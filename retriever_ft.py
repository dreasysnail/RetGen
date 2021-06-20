#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import torch
import time
import logging
import numpy as np
import os
import copy
from torch.distributed import get_rank, get_world_size
from collections import OrderedDict

from dialogpt.gpt2_training.eval_utils import eval_model_loss, eval_model_loss_joint_training, retrieve_top_docs, compare_models
from dialogpt.gpt2_training.train_utils import RedditExample
from dialogpt.data_loader import convert_examples_to_features_dynamic
from dialogpt.gpt2_training.train_utils import load_model, boolean_string, set_lr, get_eval_list_same_length
from dialogpt.gpt2_training.distributed import all_reduce_and_rescale_tensors, all_gather_list
from extract_top_docs import init_retriever, init_retriever_single_rank
from dpr.utils.model_utils import CheckpointState

logger = logging.getLogger(__name__)

def generate_str_vectors(encoder, tensorizer, questions, device, bsz=1):
    n = len(questions)
    query_vectors = []

    for j, batch_start in enumerate(range(0, n, bsz)):

        batch_token_tensors = [tensorizer.text_to_tensor(q) for q in
                               questions[batch_start:batch_start + bsz]]

        q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
        q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
        q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)
        q_ids_batch.to(device)
        q_seg_batch.to(device)
        q_attn_mask.to(device)

        _, out, _ = encoder(q_ids_batch, q_seg_batch, q_attn_mask)

        query_vectors.extend(out.cpu().split(1, dim=0))

        if len(query_vectors) % 100 == 0:
            logger.info('Encoded queries %d', len(query_vectors))

    query_tensor = torch.cat(query_vectors, dim=0).to(device)

    assert query_tensor.size(0) == len(questions)
    return query_tensor

def retriever_finetune(args, batch, eval_dataloader_loss, global_step, EPS, device, step, n_gpu,
                       enc, retriever_last, generator_model, retriever_model,
                       optimizer, tensorizer, all_passages,
                       config, epoch, pbar, train_logger, eval_logger, output_dir, stats ,experiment=None):

    generator_model.eval()
    retriever_model.train()
    step -= 1
    (tr_loss, tr_ppl, mean_ppl, nb_tr_examples, nb_tr_steps) = stats['tr_loss'], stats['tr_ppl'], stats['mean_ppl'], stats['nb_tr_examples'], stats['nb_tr_steps']
    tr_dot_product, tr_reward = stats['tr_dot_product'], stats['tr_reward']
    n_token_real, n_token_total = stats['n_token_real'], stats['n_token_total']
    train_start_time_epoch = time.time()
    ret_time, gen_time = stats['ret_time'], stats['gen_time']

    if (global_step) % args.valid_step == 0:  
        retriever_model.eval()
        if global_step != 0:
            if hasattr(retriever_model, 'module'):
                retriever_last, all_passages = init_retriever(args, eval_on_each=args.eval_on_each,
                                            encoder=copy.deepcopy(retriever_model.module.question_model), tensorizer=tensorizer,
                                            force_index=True, file_suffix = args.file_suffix)
            else:
                retriever_last, all_passages = init_retriever(args, eval_on_each=args.eval_on_each,
                                            encoder=copy.deepcopy(retriever_model.question_model), tensorizer=tensorizer,
                                            force_index=True, file_suffix = args.file_suffix)

        if args.local_rank == -1 or get_rank() == 0:
            state_dict = {k.replace('module.',''): (v.cpu() if v is not None else None)  
                    for k, v in retriever_model.state_dict().items()}  
            torch.save(OrderedDict([('model_dict', state_dict), 
                        ('optimizer_dict', None),
                        ('scheduler_dict', None),
                        ('offset', None),
                        ('epoch', None),
                        ('encoder_params', None)]),
                    os.path.join(output_dir,
                    f'retriever-pretrain-step-{global_step}.pkl'))
            eval_loss, eval_ppl, eval_reward = eval_model_loss_joint_training(
                generator_model, retriever_last, all_passages, enc, eval_dataloader_loss, epoch, args)
            print('r step:{},eval_loss:{},eval_ppl:{},eval_reward:{}'.format(global_step+1, eval_loss, eval_ppl, eval_reward), flush=True)
            print('R,{},{},{},{},{},{}'.format(
                epoch+1, global_step+1, step+1, eval_loss, eval_ppl, eval_reward),
                file=eval_logger, flush=True)
            if experiment is not None:
                experiment.log_metrics({
                    'epoch_ret': epoch + 1,
                    'global_step_ret': global_step + 1,
                    'step_ret': step + 1,
                    'eval_loss_ret': eval_loss,
                    'eval_ppl_ret': eval_ppl,
                    'eval_reward_ret': eval_reward
                })

            logger.info('current learning rate: '
                        + str(optimizer.param_groups[0]['lr']))

        if hasattr(retriever_model, 'module') and n_gpu>1:
            torch.distributed.barrier()

        retriever_model.train()

    seq_len = batch[0].shape[1]
    batch = tuple(t.to(args.device) for t in batch)
    input_ids, position_ids, token_ids, label_ids, *_ = batch

    ret_start_time = time.time()
    SUMMATION_RANGE = 1

    with torch.no_grad():
        args.n_docs *= SUMMATION_RANGE

        ret_passages, ret_scores, cxt_str, rsp_str = retrieve_top_docs(input_ids, enc, retriever_last, all_passages, args)
        args.n_docs = int(args.n_docs / SUMMATION_RANGE)

        ret_end_time = time.time()
        loss_ret_topK = []
        all_example = []
        for t in range(args.n_docs * SUMMATION_RANGE): 

            doc_lines = [' '.join(doc.strip().split()) for doc in ret_passages[t]]
            examples = [RedditExample(i, doc_line, src_line, tgt_line) for i, (doc_line, src_line, tgt_line) in
                        enumerate(zip(doc_lines, cxt_str, rsp_str))]
            features = convert_examples_to_features_dynamic(examples, enc,
                                                            args.max_seq_length)  
            batch_ret = eval_dataloader_loss._batch_feature(features)
            batch_ret = tuple(t.to(args.device) for t in batch_ret)
            input_ids_ret, position_ids_ret, token_ids_ret, label_ids_ret, *_ = batch_ret
            loss_ret, _ = generator_model.forward_pointwise(input_ids_ret, position_ids_ret, token_ids_ret,
                                                            label_ids_ret)
            loss_ret_topK.append(loss_ret)   
            all_example.extend(examples)

    coeff = torch.exp(-torch.stack(loss_ret_topK))  

    if hasattr(retriever_model, 'module'):
        query_vector = generate_str_vectors(retriever_model.module.question_model, tensorizer, cxt_str, args.device)  
        psg_vector = torch.stack([generate_str_vectors(retriever_model.module.ctx_model, tensorizer,
                                                        [ret_passages[i][j] for i in range(args.n_docs)], args.device)
                                    for j in range(len(ret_passages[0]))])  
    else:
        query_vector = generate_str_vectors(retriever_model.question_model, tensorizer, cxt_str, args.device, args.train_batch_size)  
        psg_vector = torch.stack([generate_str_vectors(retriever_model.ctx_model, tensorizer,
                                                        [ret_passages[i][j] for i in range(args.n_docs)], args.device, args.train_batch_size)
                                    for j in range(len(ret_passages[0]))])  

    mapping_dim = psg_vector.shape[-1]
    dot_product = torch.stack([torch.mv(psg_vector[i], query_vector[i]) for i in range(len(query_vector))], dim=1) 

    with torch.no_grad():

        d_scores = dot_product + EPS 
        d_scores = d_scores - torch.mean(d_scores, axis=0)

        normalized_score = torch.softmax(d_scores.to(device), dim=0)  

    if args.rl_method == "simple":
        reward = coeff  
        reward = reward - torch.mean(reward, 0)
        logpz_x = torch.log_softmax(dot_product, dim = 0)
        loss = -logpz_x * normalized_score* reward  
    else:
        raise NotImplementedError('rl method cannot be ' + args.rl_method)

    loss = loss.sum(0).mean()
    dot_product = dot_product.sum(0).mean()
    reward = reward.sum(0).mean()

    if n_gpu > 1:
        loss = loss.mean()
        dot_product = dot_product.mean()
        reward = reward.mean()

    if args.fp16:
        from apex import amp

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm=1.0)
    else:
        loss.backward()

    tr_loss += float(loss.item()) * (args.train_batch_size / input_ids.shape[0])
    tr_dot_product += float(dot_product.item()) * (args.train_batch_size / input_ids.shape[0])

    tr_reward += torch.exp(-torch.stack(loss_ret_topK)).mean().item() * input_ids.size(0)

    nb_tr_examples += input_ids.size(0)
    nb_tr_steps += 1
    mean_loss = tr_loss / nb_tr_steps
    mean_dot_product = tr_dot_product / nb_tr_steps
    mean_reward = tr_reward / nb_tr_steps

    n_token_total += input_ids.shape[0] * input_ids.shape[1]
    n_token_real += (input_ids != 0).sum().item()
    gen_end_time = time.time()
    ret_time += (ret_end_time - ret_start_time)
    gen_time += (gen_end_time - ret_end_time)

    step += 1
    if step % args.gradient_accumulation_steps == 0:
        global_step -= 1
        set_lr(optimizer, global_step,
                args.lr_schedule, args.learning_rate,
                args.warmup_steps, args.warmup_proportion,
                config.n_embd, args.num_optim_steps)

        if args.local_rank != -1:
            grads = [p.grad.data for p in retriever_model.parameters()
                        if p.requires_grad and p.grad is not None]
            all_reduce_and_rescale_tensors(grads, float(1))  

        optimizer.step()
        global_step += 1

        retriever_model.zero_grad()

        if args.local_rank != -1:
            mean_loss = sum(all_gather_list(mean_loss)) / get_world_size()
            mean_dot_product = sum(all_gather_list(mean_dot_product)) / get_world_size()
            mean_reward = sum(all_gather_list(mean_reward)) / get_world_size()
            mean_ppl = sum(all_gather_list(mean_ppl)) / get_world_size()
            n_token_real_all_proc = sum(all_gather_list(n_token_real))
            n_token_total_all_proc = sum(all_gather_list(n_token_total))
        else:
            n_token_real_all_proc = n_token_real
            n_token_total_all_proc = n_token_total

        if args.local_rank == -1 or get_rank() == 0:
            epoch_time = time.time() - train_start_time_epoch
            if pbar is not None:
                pbar.set_postfix_str(
                    f"tok/s: {n_token_real_all_proc // epoch_time // 1000}k "
                    f"ppl: {mean_ppl:.2f}   loss:{mean_loss:.2f}   epoch: {epoch}")
                pbar.update(1)
            print('Epoch:{}, Retrival step:{},loss:{:.3f},ppl:{:.3f},ret_time:{:.3f},gen_time:{:.3f},reward:{:.3f},dot_prod:{:.3f}'.format(epoch + 1, global_step + 1, mean_loss, mean_ppl, ret_time, gen_time, mean_reward, mean_dot_product), flush=True)
            print('R,{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}'.format(
                epoch + 1, global_step + 1, mean_loss, mean_ppl, ret_time, gen_time, mean_reward, mean_dot_product),
                file=train_logger, flush=True)
            ret_time, gen_time = 0.0, 0.0

    generator_model.train()

    stats['tr_loss'], stats['tr_ppl'], stats['mean_ppl'], stats['nb_tr_examples'], stats['nb_tr_steps'] = (tr_loss, tr_ppl, mean_ppl, nb_tr_examples, nb_tr_steps) 
    stats['tr_dot_product'], stats['tr_reward'] = tr_dot_product, tr_reward
    stats['n_token_real'], stats['n_token_total'] = n_token_real, n_token_total
    stats['ret_time'], stats['gen_time'] = ret_time, gen_time
    return stats

