import json
from os.path import abspath, dirname, exists, join
import argparse
import logging
from tqdm import trange
import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import socket
import os, sys
import re
import time
from pprint import pprint

from env import PROJECT_FOLDER

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

from gpt2_training.train_utils import get_eval_list_same_length, load_model, boolean_string, get_eval_list_same_length_with_order, fix_state_dict_namespace
from gpt2_training.generation import generate_sequence, cut_seq_to_eos, beam_search_naive, grid_beam_search
from gpt2_training.eval_utils import cal_entropy, cal_BLEU_4, cut_seq_to_eos, EOS_ID
import pdb

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output

def run_model():
    print(socket.gethostname())

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='/philly/sc3/resrchvc/yizzhang/GPT/pretrained/117M', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_checkpoint", '-c', type=str, default='/philly/sc3/resrchvc/yizzhang/GPT/pretrained/117M/pytorch_model.bin')
    parser.add_argument("--fp16", type=boolean_string, default=False)
    parser.add_argument("--test_file", '-t', type=str, default=None, help='input file for testing')

    parser.add_argument("--normalize_data", type=boolean_string, default=True)
    parser.add_argument("--batch_size", '-b', type=int, default=256)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--no_token_id", action='store_true')
    parser.add_argument("--no_eos", action='store_true')

    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--config', help='JSON config file')
    parser.add_argument("--rev_model_checkpoint", type=str, default="../Dialogpt_dev_data/small_reverse.pkl")
    parser.add_argument("--output_file", '-o', type=str, default=None, help='output file for testing')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

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

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu
    pprint(args)

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    config = GPT2Config.from_json_file(os.path.join(args.model_name_or_path, 'config.json'))
    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    model = load_model(GPT2LMHeadModel(config), args.load_checkpoint, args, verbose=True)
    model.to(device)
    model.eval()

    rev_model = GPT2LMHeadModel(config)
    model_state_dict = fix_state_dict_namespace(torch.load(args.rev_model_checkpoint))
    start_model = rev_model
    if hasattr(rev_model, "transformer") and all(not s.startswith('transformer.') for s in model_state_dict.keys()):
        print('loading transfomer only')
        start_model = rev_model.transformer
    start_model.load_state_dict(model_state_dict)
    if args.fp16:
        rev_model.half()
    rev_model.to(device)
    rev_model.eval()

    eval_dataloader = get_eval_list_same_length_with_order(args.test_file, enc, args.batch_size, True, for_eval=True)
    model.eval()
    outs = []
    targets = []
    loss_all = []
    ppl_all = []
    rev_ppl_all = []
    sources = []
    conv_ids = []
    with torch.no_grad():
        with tqdm.tqdm(total=len(eval_dataloader), desc=f"Test") as pbar:
            for step, batch in enumerate(tqdm.tqdm(eval_dataloader, desc="Iteration")):
                new_batch = []
                for t in batch:
                    if isinstance(t,list):
                        new_batch.append(t)
                    else:
                        new_batch.append(t.to(device))
                input_ids, position_ids, token_ids,  label_ids, doc_len, context_len, target_len, conv_id = new_batch

                if args.no_token_id:
                    token_ids = None
                if args.no_eos:
                    input_ids = input_ids[:,:-1]

                def cal_ppl(input_ids, label_ids, model, rev=False):
                    if rev:

                        lab_end_idx = [np.where(o.cpu().numpy()==EOS_ID)[0] for o in label_ids]
                        lab_end_idx = [o[-2] if len(o)>1 else -1 for o in lab_end_idx]
                        _label_ids = torch.stack([torch.cat([torch.ones_like(src[:-1]) *-1 , lab[idx+1:], torch.ones_like(lab[:idx+1]).view(-1) *-1, torch.ones_like(src[-1]).view(-1) *-1]) for src, lab, idx in zip(input_ids, label_ids, lab_end_idx)])
                        _input_ids = torch.stack([torch.cat([src, lab[idx+1:], lab[:idx+1]]) for src, lab, idx in zip(input_ids, label_ids, lab_end_idx)])

                    else:
                        _label_ids = torch.stack([torch.cat([torch.ones_like(src[:-1]) *-1 , lab , torch.ones_like(src[-1]).view(-1) *-1]) for src, lab in zip(input_ids, label_ids)])
                        _input_ids = torch.stack([torch.cat([src, lab]) for src, lab in zip(input_ids, label_ids)])
                    _input_ids= torch.nn.utils.rnn.pad_sequence([torch.tensor(o, dtype=torch.long).to(device) for o in _input_ids], batch_first=True, padding_value=EOS_ID)
                    _label_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(o, dtype=torch.long).to(device) for o in _label_ids], batch_first=True, padding_value=-1)

                    _, _ppl = model.forward_pointwise(_input_ids, None, None, _label_ids)
                    return _ppl
                ppl = cal_ppl(input_ids, label_ids, model)
                rev_ppl = cal_ppl(label_ids, input_ids, rev_model,rev=True)

                sources.extend([i.cpu().numpy() for i in input_ids])
                targets.extend([i.cpu().numpy() for i in label_ids])
                conv_ids.extend(conv_id.cpu().numpy())
                ppl_all.extend(ppl.cpu().numpy())
                rev_ppl_all.extend(rev_ppl.cpu().numpy())
                torch.cuda.empty_cache()

            conv_id_map = {conv_ids[i]: i for i in range(len(conv_ids))}

            src = [enc.decode(s).encode('ascii','ignore').decode('ascii') for s in sources]

            tgt = [enc.decode(s).encode('ascii','ignore').decode('ascii') for s in targets]

            src_orders = [src[conv_id_map[i]] for i in sorted(conv_id_map)]
            tgt_orders = [tgt[conv_id_map[i]] for i in sorted(conv_id_map)]
            ppl_orders = [ppl_all[conv_id_map[i]] for i in sorted(conv_id_map)]
            rev_ppl_orders = [rev_ppl_all[conv_id_map[i]] for i in sorted(conv_id_map)]

            with open(args.test_file[:-3] + (args.output_file + '.' if args.output_file else '') + 'eval.txt', "w") as eval_f:
                eval_f.write(f"Source\tTarget\tPPL\tRev_PPL\n")
                for i,r in enumerate(src_orders):
                    r = re.sub("\n", "", r)
                    eval_f.write(f"{src_orders[i]}\t{tgt_orders[i]}\t{ppl_orders[i]:.3f}\t{rev_ppl_orders[i]:.3f}\n")

            sys.stdout.flush()

if __name__ == '__main__':
    run_model()

