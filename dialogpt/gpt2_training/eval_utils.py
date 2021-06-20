'''
* @Date: 2019-04-02 13:46:04
'''
import sys
import torch
import tqdm
import logging

import numpy as np

from torch.distributed import get_rank, get_world_size
from pycocoevalcap.bleu.bleu import Bleu
from collections import OrderedDict
from pycocoevalcap.rouge.rouge import Rouge
from pdb import set_trace as bp
from collections import defaultdict

from gpt2_training.generation import generate_sequence, cut_seq_to_eos
sys.path.append("..")
from env import TOKEN_TYPE_CXT, TOKEN_TYPE_DOC, EOS_ID, PAD_ID
from data_loader import convert_examples_to_features_dynamic
from gpt2_training.train_utils import RedditExample
logger = logging.getLogger(__name__)
import re
from gpt2_training.distributed import all_reduce_and_rescale_tensors, all_gather_list

def cal_BLEU_4(generated, reference, is_corpus = False):
    BLEUscore = [0.0,0.0,0.0,0.0]
    for idx, g in enumerate(generated):
        if is_corpus:
            score, scores = Bleu(4).compute_score(reference, {0: [g]})
        else:
            score, scores = Bleu(4).compute_score({0: [reference[0][idx]]} , {0: [g]})
        for i, s in zip([0,1,2,3],score):
            BLEUscore[i]+=s
    BLEUscore[0] = BLEUscore[0]/len(generated)
    BLEUscore[1] = BLEUscore[1]/len(generated)
    BLEUscore[2] = BLEUscore[2]/len(generated)
    BLEUscore[3] = BLEUscore[3]/len(generated)
    return BLEUscore

def cal_entropy(generated):
    etp_score = [0.0,0.0,0.0,0.0]
    div_score = [0.0,0.0,0.0,0.0]
    counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) +1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) /total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) /total
    return etp_score, div_score

def eval_model_generation(model, tokenizer, eval_dataloader, epoch_id, args, use_beam_search=False, beam_width=3):
    model.eval()
    outs = []
    targets = []
    sources = []
    with torch.no_grad():
        with tqdm.tqdm(total=len(eval_dataloader), desc=f"Epoch {epoch_id-1} dev set", miniters=int(len(eval_dataloader)/10)) as pbar:
            for step, batch in enumerate(eval_dataloader):
                new_batch = []
                for t in batch:
                    if isinstance(t,list):
                        new_batch.append(t)
                    else:
                        new_batch.append(t.to(args.device))
                input_ids, position_ids, token_ids,  label_ids, src_len, _ = new_batch
                if not args.no_token_id:
                    new_token_ids = []
                    tot_len = input_ids.size(1)
                    for s in src_len:
                        new_token_ids.append(torch.cat((torch.zeros([1,s], dtype=token_ids.dtype, device=token_ids.device), torch.ones([1,tot_len - s], dtype=token_ids.dtype, device=token_ids.device)),1 ) )
                    token_ids = torch.stack(new_token_ids, dim=1)
                if args.no_token_id:
                    token_ids = None

            
                out = generate_sequence(model, input_ids, position_ids, token_ids,
                                        length=args.generation_length,
                                        start_token=None,
                                        temperature=args.temperature, top_k=args.top_k,
                                        sample=args.is_sampling)
                sources.extend(input_ids.cpu().numpy())
                out = out.tolist()
                outs.extend(out)
                target = [[x.cpu().numpy() for x in l if x != -1] for l in label_ids]
                if isinstance(target[0][0], np.ndarray):
                    target = [[t.item() for t in tt] for tt in target]
                targets.extend(target)
            val_src = [tokenizer.decode(cut_seq_to_eos(s)).encode('utf-8').decode('utf-8').strip() for s in sources]
            val_set = [tokenizer.decode(cut_seq_to_eos(s)).encode('utf-8').decode('utf-8').strip() for s in targets]
            gen = [tokenizer.decode(cut_seq_to_eos(s)).encode('utf-8').decode('utf-8').strip() for s in outs]
            [bleu1s,bleu2s,bleu3s,bleu4s] = cal_BLEU_4(gen, {0: val_set}, is_corpus = False)
            etp_score, dist_score = cal_entropy(gen)

            print("=" * 80)
            print ("")
            print('Val BLEU: ' + ' '.join([str(round(it,3)) for it in (bleu1s,bleu2s,bleu3s,bleu4s)]))
            print('Val Entropy: ' + ' '.join([str(round(it,3)) for it in (etp_score[0],etp_score[1],etp_score[2],etp_score[3])]))
            print('Val Diversity: ' + ' '.join([str(round(it,3)) for it in (dist_score[0],dist_score[1],dist_score[2],dist_score[3])]))
            for n_s in range(2,args.nsamples):
                print("=" * 40 + " SAMPLE " + str(n_s) + "=" * 40)
                src = val_src[-1-n_s*50]
                gt = val_set[-1-n_s*50]
                resp = gen[-1-n_s*50]
                print(f"Source: \t {src} \n Oracle: \t {gt} \n Resp: \t {resp}\n")
                print ("")
                print("=" * 80)

            sys.stdout.flush()
            torch.cuda.empty_cache()
            return gen

def eval_model_loss(model, tokenizer, eval_dataloader, epoch_id, args):
    logger.info('compute eval model loss, using eval mode, please change it back to train after calling this function')
    model.eval()
    tot_loss = []
    tot_ppl = []
    tot_sample = []
    model_ = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, doc_len, src_len, _ = batch


            if args.no_token_id:
                token_ids = None
            n_sample = input_ids.shape[0]

            loss, ppl = model_(input_ids, position_ids, token_ids, label_ids)
            tot_loss.append(loss.mean().item() * n_sample)
            tot_ppl.append(ppl.mean().item() * n_sample)
            tot_sample.append(n_sample)
    print(f"\n Epoch {epoch_id}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)} Val ppl {np.sum(tot_ppl) / np.sum(tot_sample)} ")
    return np.sum(tot_loss) / np.sum(tot_sample), np.sum(tot_ppl) / np.sum(tot_sample)

def eval_model_loss_joint_training(model, retriever, all_passages, enc, eval_dataloader, epoch_id, args):
    logger.info('compute eval model loss, using eval mode, please change it back to train after calling this function')
    model.eval()
    tot_loss = []
    tot_ppl = []
    tot_sample = []
    tot_reward = []
    model_ = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, doc_len, src_len, _ = batch

            ret_passages, ret_scores, cxt_str, rsp_str = retrieve_top_docs(input_ids, enc, retriever, all_passages, args)
            loss_ret_topK = []
            for t in range(args.n_docs):

                doc_lines = [' '.join(doc.strip().split()) for doc in ret_passages[t]]
                examples = [RedditExample(i, doc_line, src_line, tgt_line) for i,(doc_line, src_line, tgt_line) in enumerate(zip(doc_lines, cxt_str, rsp_str))]
                features = convert_examples_to_features_dynamic(examples, enc, args.max_seq_length)
                batch_ret = eval_dataloader._batch_feature(features)
                batch_ret = tuple(t.to(args.device) for t in batch_ret)
                input_ids_ret, position_ids_ret, token_ids_ret, label_ids_ret, *_ = batch_ret 
                loss_ret, _ = model_.forward_pointwise(input_ids_ret, position_ids_ret, token_ids_ret, label_ids_ret)
                loss_ret_topK.append(loss_ret)

            normalized_score = torch.softmax(torch.tensor(ret_scores).to(args.device),dim = 0)
            if args.avg_by_prob:
                loss = - torch.mean(torch.logsumexp( - torch.stack(loss_ret_topK, dim = 0) + torch.log_softmax(torch.tensor(ret_scores).to(device),dim = 0), dim = 0))
            else:
                loss = torch.mean(torch.sum(normalized_score * torch.stack(loss_ret_topK, dim = 0), dim = 0))

            ppl = torch.exp(loss)
            n_sample = input_ids.shape[0]            



            tot_loss.append(loss.mean().item() * n_sample)
            tot_ppl.append(ppl.mean().item() * n_sample)
            tot_reward.append(torch.exp(-torch.stack(loss_ret_topK)).mean().item() * n_sample)

            tot_sample.append(n_sample)


    print(f"\n Epoch {epoch_id}: Val loss {np.sum(tot_loss) / np.sum(tot_sample):.3f} Val ppl {np.sum(tot_ppl) / np.sum(tot_sample):.3f} Val reward {np.sum(tot_reward) / np.sum(tot_sample):.3f}")

    return np.sum(tot_loss) / np.sum(tot_sample), np.sum(tot_ppl) / np.sum(tot_sample), np.sum(tot_reward) / np.sum(tot_sample)

def retrieve_top_docs(input_ids, enc, retriever, all_passages, args):
    """
    docstring
    """
    with torch.no_grad():
        input_ids_cpu = input_ids.cpu().numpy()
        start_idx = [np.where(o==EOS_ID)[0][0] for o in input_ids_cpu]
        end_idx = [np.where(o==EOS_ID)[0][-1] for o in input_ids_cpu]
        def find_pad_start(input_ids_cpu):
            pad_idx = [None] *len(input_ids_cpu)
            i = len(input_ids_cpu[0]) - 1
            rows = set([idx for idx, x in enumerate(input_ids_cpu[:,i]) if x != 0])
            all_rows = set(list(range(len(input_ids_cpu))))
            while i >= -1:
                remain_rows = all_rows - rows
                if len(remain_rows) == 0:
                    break
                for r in remain_rows.copy():
                    if input_ids_cpu[r,i] != 0 or i==-1:
                        rows.add(r)
                        pad_idx[r] = i+1
                i -= 1
            return pad_idx

        pad_idx = find_pad_start(input_ids_cpu)

        cxt = [o[s+1:e] for o, s, e in zip(list(input_ids_cpu),start_idx,end_idx)]
        cxt_str = [enc.decode(c).encode('ascii','ignore').decode('ascii') for c in cxt]
        cxt_str = [re.sub('<\|endoftext\|>', ' EOS ',c) for c in cxt_str]

        rsp = [o[s+1:e] for o, s, e in zip(list(input_ids_cpu),end_idx,pad_idx)]
        rsp_str = [enc.decode(r).encode('ascii','ignore').decode('ascii') for r in rsp]

        qry_str = [" " + c for c in cxt_str]
        questions_tensor = retriever.generate_question_vectors(qry_str)
        top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs, is_hnsw = args.hnsw_index)
        # save top docs and scores
        ret_passages = list(zip(*[[all_passages[str(idx).strip()][0] for idx in it[0]] for it in top_ids_and_scores]))
        ret_scores = list(zip(*[[float(s) for s in it[1]] for it in top_ids_and_scores]))
    return ret_passages, ret_scores, qry_str, rsp_str

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

