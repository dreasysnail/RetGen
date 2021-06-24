#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
'''
 * @Desc: Retriever + generator joint training
'''
import torch
import os

import json
import copy
import sys
import argparse
import logging
import time
import tqdm
import datetime
from collections import defaultdict

sys.path.append("./dialogpt")
import numpy as np

from os.path import join
from torch.distributed import get_rank, get_world_size
from dialogpt.env import TOKEN_TYPE_CXT, TOKEN_TYPE_DOC, EOS_ID, PAD_ID
from dialogpt.lsp_model import Adam
from dialogpt.pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from dialogpt.gpt2_training.train_utils import load_model, boolean_string, set_lr, get_eval_list_same_length
from dialogpt.gpt2_training.eval_utils import eval_model_loss, eval_model_loss_joint_training, retrieve_top_docs, compare_models

from dialogpt.data_loader import BucketingDataLoader, DynamicBatchingLoader, DistributedBucketingDataLoader

from dialogpt.gpt2_training.distributed import all_reduce_and_rescale_tensors, all_gather_list

from extract_top_docs import init_retriever, init_retriever_single_rank
from dpr.options import add_encoder_params, add_cuda_params, setup_args_gpu, set_seed, print_args, add_training_params
from dpr.utils.model_utils import load_states_from_checkpoint_only_model

from dense_retriever import *
import re
from dialogpt.gpt2_training.train_utils import RedditExample
from dialogpt.data_loader import convert_examples_to_features_dynamic
from retriever_ft import generate_str_vectors, retriever_finetune

EPS = 1e-6
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

INF = 100000000
CACHE_EMPTY_STEP = 10000
EVAL_STEP = 100000

parser = argparse.ArgumentParser()

add_encoder_params(parser)
parser.add_argument("--adam_eps", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="")

parser.add_argument('--model_name_or_path', type=str,
                    help='pretrained model name or path to local checkpoint')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_seq_length", type=int, default=512)

parser.add_argument("--skip_eval", action='store_true',
                    help='If true, skip evaluation.')
parser.add_argument("--init_checkpoint", type=str)
parser.add_argument("--train_input_file", type=str)
parser.add_argument("--eval_input_file", type=str)
parser.add_argument("--continue_from", type=int, default=0)

parser.add_argument("--train_batch_size", type=int, default=4,
                    help="batch size now means per GPU per step")
parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                    help="to increase effective batch size "
                         "and reduce synchronization")
parser.add_argument("--eval_batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=1e-6)
parser.add_argument("--num_optim_steps", type=int, default=1000000,
                    help="new API specifies num update steps")
parser.add_argument("--valid_step", type=int, default=200,
                    help="how many optim steps between validations")
parser.add_argument("--print_step", type=int, default=100,
                    help="how many optim steps between validations")
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--warmup_steps", type=int, default=1)

parser.add_argument("--normalize_data", type=boolean_string, default=True)
parser.add_argument("--lr_schedule", type=str,
                    choices=['noam', 'noamwd', 'BERT', 'None'], default='noam')
parser.add_argument("--loss_scale", type=float, default=0)
parser.add_argument("--no_token_id", type=boolean_string, default=False)

parser.add_argument("--output_dir", type=str)
parser.add_argument("--log_dir", type=str)
parser.add_argument('--pbar', type=boolean_string, default=False, help='turn on progress bar')
parser.add_argument('--set_type_embedding_to_zero', type=boolean_string, default=False, help='initialize type embedding to zero for generator')
parser.add_argument('--reverse', type=boolean_string, default=False, help='reverse training')
parser.add_argument('--avg_by_prob', type=boolean_string, default=False, help='average the loss by prob')
parser.add_argument('--rl_method', help='RL method', default="simple", choices=['simple'])
parser.add_argument('--file_suffix', help='file suffix to save doc embedding and index', default="fine_tune")
parser.add_argument("--dropout", default=0.1, type=float, help="")

parser.add_argument('--config', help='JSON config file')

add_cuda_params(parser)

parser.add_argument('--ctx_file', type=str, default=None, help='Path to passages set .tsv file for encoding')
parser.add_argument('--shard_id', type=int, default=0, help="Number(0-based) of data shard to process")
parser.add_argument('--num_shards', type=int, default=1, help="Total amount of data shards")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
parser.add_argument('--n_docs', type=int, default=5, help="Amount of top docs to return")
parser.add_argument('--validation_workers', type=int, default=16,
                    help="Number of parallel processes to validate results")
parser.add_argument('--index_buffer', type=int, default=50000,
                    help="Temporal memory data buffer size (in samples) for indexer")
parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
parser.add_argument("--encoding", action='store_true', help='If enabled, encode the doc features offline')
parser.add_argument("--shard_folder", action='store_true', help='If enabled, encode the doc features are sharded')
parser.add_argument("--log_batch_step", default=100, type=int, help="")
parser.add_argument("--eval_on_each", action='store_true', help='If enabled, eval on each shard')
parser.add_argument("--retriever_master_rank", type=boolean_string, default=False, help='not for single card')  

parser.add_argument("--r_only", action='store_true', help='only r')
parser.add_argument("--g_only", action='store_true', help='only g')
parser.add_argument("--ret_correction", action='store_true', help='retriver correction')
parser.add_argument("--change_id_over_card", type=boolean_string, default=True, help='change shard id over card')

parser.add_argument("--load_trained_model", action='store_true', help='If enabled, load trained retrieval model')

args = parser.parse_args()
args.do_lower_case = True
if args.encoding:
    args.file_suffix = ""

if args.config is not None:

    opts = json.load(open(args.config))
    for k, v in opts.items():
        if isinstance(v, str):

            if 'PHILLY_JOB_DIRECTORY' in v:
                v = v.replace('PHILLY_JOB_DIRECTORY',
                              os.environ['PHILLY_JOB_DIRECTORY'])
            elif 'PHILLY_LOG_DIRECTORY' in v:
                v = v.replace('PHILLY_LOG_DIRECTORY',
                              os.environ['PHILLY_LOG_DIRECTORY'])
        setattr(args, k, v)

    argv = sys.argv[1:]
    overrides, _ = parser.parse_known_args(argv)
    for k, v in vars(overrides).items():
        if f'--{k}' in argv:
            setattr(args, k, v)
    setattr(args, 'local_rank', overrides.local_rank)

assert args.train_batch_size % args.gradient_accumulation_steps == 0, 'batch size % gradient accumulation steps != 0!'
args.train_batch_size = (args.train_batch_size // args.gradient_accumulation_steps)
logger.info('train batch size = {}, '
            'new train batch size (after gradient accumulation) = {}'.format(
                args.train_batch_size*args.gradient_accumulation_steps,
                args.train_batch_size))

if args.local_rank == -1:
    logger.info('CUDA available? {}'.format(str(torch.cuda.is_available())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu
else:

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    torch.distributed.init_process_group(backend='nccl')
    n_gpu = torch.distributed.get_world_size()
    args.device, args.n_gpu = device, 1
    logger.info("device: {} n_gpu: {}, distributed training: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, bool(args.local_rank != -1), args.fp16))

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
output_dir = join(args.output_dir,
                  'GPT2.{}.{}.{}gpu.{}'.format(args.learning_rate,
                                               args.train_batch_size, n_gpu,
                                               timestamp))
log_dir = args.log_dir if args.log_dir is not None and len(args.log_dir) > 0 else output_dir
if args.local_rank == -1 or get_rank() == 0:
    experiment = None
    os.makedirs(output_dir, exist_ok=True)
else:
    experiment = None

logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)

config = GPT2Config.from_json_file(
    join(args.model_name_or_path, 'config.json'))

if args.local_rank == -1:
    train_dataloader = BucketingDataLoader(args.train_input_file,
                                           args.train_batch_size,
                                           args.max_seq_length,
                                           shuffle = True)  
else:
    train_dataloader = DistributedBucketingDataLoader(
        get_rank(), get_world_size(),
        args.train_input_file, args.train_batch_size,
        args.max_seq_length)

eval_dataloader_loss = DynamicBatchingLoader(
    args.eval_input_file, enc, args.normalize_data,
    args.eval_batch_size, args.max_seq_length, reverse = args.reverse)

set_seed(args)
setup_args_gpu(args)
args.do_lower_case = True

tensorizer, bi_encoder, optimizer_retriever = init_biencoder_components(args.encoder_model_type, args)
if args.model_file:
    if not args.load_trained_model:
        saved_state = load_states_from_checkpoint(args.model_file)
    else: 
        saved_state = load_states_from_checkpoint_only_model(args.model_file)
    bi_encoder.load_state_dict(saved_state.model_dict)

retriever_infer, all_passages = init_retriever(args, eval_on_each=args.eval_on_each,
                                               encoder=copy.deepcopy(bi_encoder.question_model), tensorizer=tensorizer,
                                               force_index=False)

bi_encoder, optimizer_retriever = setup_for_distributed_mode(bi_encoder, optimizer_retriever, args.device, args.n_gpu,
                                                             args.local_rank,
                                                             args.fp16,
                                                             args.fp16_opt_level)

model = load_model(GPT2LMHeadModel(config, tied=False), args.init_checkpoint,
                   args, verbose=True, set_type_embedding_to_zero=args.set_type_embedding_to_zero)

if args.local_rank != -1:

    params = [p.data for p in model.parameters()]
    all_reduce_and_rescale_tensors(
        params, float(torch.distributed.get_world_size()))

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
total_params = sum([np.prod(p.size()) for p in model_parameters])
logger.info('Number of parameter = {}'.format(total_params))

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'ln']   
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = Adam(optimizer_grouped_parameters, args.learning_rate, max_grad_norm=1.0)

if args.fp16:
    logger.info('in fp16, using FusedAdam')
    try:
        import apex
        from apex import amp
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex "
            "to use distributed and fp16 training.")

    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

if args.local_rank == -1 or get_rank() == 0:
    train_logger = open(join(log_dir, 'train_log.txt'), 'a+', buffering=-1)
    train_r_logger = open(join(log_dir, 'train_ret_log.txt'), 'a+', buffering=-1)
    eval_logger = open(join(log_dir, 'eval_log.txt'), 'a+', buffering=-1)
    print('epoch,global_step,step,mean_loss,mean_ppl,n_token_real,'
          'n_token_total,epoch_time,ret_time,gen_time', file=train_logger)
    print('epoch,global_step,mean_loss,mean_ppl,ret_time,gen_time'
          'reward,dot_product', file=train_r_logger)
    print('epoch,global_step,step,eval_loss,eval_ppl,eval_reward', file=eval_logger)
else:
    train_logger, train_r_logger, eval_logger = None, None, None

global_step = 0
step = 0
epoch = 0

if args.continue_from:
    global_step = args.continue_from
    step = global_step*2 - 1

if args.local_rank != -1:
    n_gpu = 1
if args.local_rank == -1 or get_rank() == 0:
    if args.pbar:
        pbar = tqdm.tqdm(total=args.num_optim_steps, desc=f"training")
    else:
        pbar = None
else:
    pbar = None

if n_gpu > 1:
    model = torch.nn.DataParallel(model) 

while True:

    model.train()
    stats = defaultdict(float)

    (tr_loss, tr_ppl, mean_ppl, nb_tr_examples, nb_tr_steps) = 0.0, 0.0, 0.0, 0, 0
    n_token_real, n_token_total = 0, 0
    train_start_time_epoch = time.time()
    ret_time, gen_time = 0.0, 0.0
    ss, nn = 0.0, 0.0
    for batch in train_dataloader:
        if not args.r_only:
            if (global_step) % args.valid_step == 0:
                if args.local_rank not in [-1, 0]:
                    torch.distributed.barrier()
                if args.local_rank == -1 or get_rank() == 0:

                    torch.save(
                        {k: (v.cpu() if v is not None else None)  
                            for k, v in model.state_dict().items()},
                        join(output_dir,
                                f'generator-pretrain-step-{global_step}.pkl'))

                    eval_loss, eval_ppl, eval_reward = eval_model_loss_joint_training(
                        model, retriever_infer, all_passages, enc, eval_dataloader_loss, epoch, args)

                    print('g step:{},eval_loss:{},eval_ppl:{},eval_reward:{}'.format(global_step+1, eval_loss, eval_ppl, eval_reward), flush=True)
                    print('G,{},{},{},{},{},{}'.format(
                        epoch+1, global_step+1, step+1, eval_loss, eval_ppl, eval_reward),
                        file=eval_logger, flush=True)

                    logger.info('current learning rate: '
                                + str(optimizer.param_groups[0]['lr']))
                    model.train()
                    if args.local_rank!=-1 and get_rank() == 0:
                        torch.distributed.barrier()

            seq_len = batch[0].shape[1]
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, *_ = batch

            if args.no_token_id:
                token_ids = None

            ret_start_time = time.time()
            ret_passages, ret_scores, cxt_str, rsp_str = retrieve_top_docs(input_ids, enc, retriever_infer, all_passages, args)

            ret_end_time = time.time()
            loss_ret_topK = []
            model_ = model.module if hasattr(model, "module") else model
            for t in range(args.n_docs):

                doc_lines = [' '.join(doc.strip().split()) for doc in ret_passages[t]]
                examples = [RedditExample(i, doc_line, src_line, tgt_line) for i,(doc_line, src_line, tgt_line) in enumerate(zip(doc_lines, cxt_str, rsp_str))]
                features = convert_examples_to_features_dynamic(examples, enc, args.max_seq_length) 
                batch_ret = eval_dataloader_loss._batch_feature(features)
                batch_ret = tuple(t.to(args.device) for t in batch_ret)
                input_ids_ret, position_ids_ret, token_ids_ret, label_ids_ret, *_ = batch_ret 
                loss_ret, _ = model_.forward_pointwise(input_ids_ret, position_ids_ret, token_ids_ret, label_ids_ret)

                loss_ret_topK.append(loss_ret)

            ret_scores = [[rr + EPS for rr in r] for r in ret_scores]
            normalized_score = torch.softmax(torch.tensor(ret_scores).to(device),dim = 0)  

            if args.avg_by_prob:

                loss = - torch.mean(torch.logsumexp( - torch.stack(loss_ret_topK, dim = 0) + torch.log_softmax(torch.tensor(ret_scores).to(device),dim = 0), dim = 0))
            else:

                loss = torch.mean(torch.sum(normalized_score * torch.stack(loss_ret_topK, dim = 0), dim = 0))

            ppl = torch.exp(loss)

            if n_gpu > 1:
                loss = loss.mean()
                ppl = ppl.mean()

            if args.fp16:
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm=1.0)
            else:
                loss.backward()
            tr_loss += float(loss.item()) * (args.train_batch_size / input_ids.shape[0])
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            mean_loss = tr_loss / nb_tr_steps
            if ppl.item() < INF:
                tr_ppl += ppl.item()
            else:
                tr_ppl += mean_ppl
            mean_ppl = tr_ppl / nb_tr_steps

            n_token_total += input_ids.shape[0] * input_ids.shape[1]
            n_token_real += (input_ids != 0).sum().item()
            gen_end_time = time.time()
            ret_time += (ret_end_time - ret_start_time)
            gen_time += (gen_end_time - ret_end_time)

            step += 1

            if step % args.gradient_accumulation_steps == 0:
                set_lr(optimizer, global_step,
                    args.lr_schedule, args.learning_rate,
                    args.warmup_steps, args.warmup_proportion,
                    config.n_embd, args.num_optim_steps)

                if args.local_rank != -1:
                    grads = [p.grad.data for p in model.parameters()
                            if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads, float(1))   

                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank != -1:
                    mean_loss = sum(all_gather_list(mean_loss)) / get_world_size()
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
                            f"tok/s: {n_token_real_all_proc//epoch_time//1000}k "
                            f"ppl: {mean_ppl:.2f}   loss:{mean_loss:.2f}   epoch: {epoch}")
                        pbar.update(1)
                    if (global_step+1) % args.print_step == 0: 
                        print('Generation step:{},loss:{:.3f},ppl:{:.3f},ret_time:{:.3f},gen_time:{:.3f}'.format(
                        global_step+1, mean_loss, mean_ppl, ret_time, gen_time), flush=True)

                        print('{},{},{},{:.3f},{:.3f},{},{},{:.3f},{:.3f},{:.3f}'.format(
                        epoch+1, global_step+1, step+1, mean_loss, mean_ppl,
                        n_token_real_all_proc, n_token_total_all_proc, epoch_time, ret_time, gen_time),
                        file=train_logger, flush=True)
                    ret_time, gen_time = 0.0, 0.0

                if global_step >= args.num_optim_steps:
                    break
        else:
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1

        if not args.g_only:
            stats = retriever_finetune(args, batch, eval_dataloader_loss, global_step, EPS, device, step, n_gpu,
                                            enc, retriever_infer, model, bi_encoder,
                                            optimizer_retriever, tensorizer, all_passages,
                                            config, epoch, pbar, train_r_logger, eval_logger, output_dir, stats, experiment=experiment)

        if (step+1) % CACHE_EMPTY_STEP == 0:
            torch.cuda.empty_cache()

    if not args.g_only:
        retriever_infer, all_passages = init_retriever(args, eval_on_each=args.eval_on_each,
                                                    encoder=copy.deepcopy(bi_encoder.module.question_model), tensorizer=tensorizer,
                                                    force_index=True, file_suffix = args.file_suffix)

    if global_step >= args.num_optim_steps:
        break

    epoch += 1

if args.local_rank == -1 or get_rank() == 0:
    if pbar is not None:
        pbar.close()
    train_logger.close()
    train_r_logger.close()
    eval_logger.close()
