"""
preprocess input data into feature and stores binary as python shelve DB
each chunk is gzipped JSON string
"""
import argparse
import gzip
import json
import subprocess as sp
import shelve
import os
from os.path import dirname, exists, join

import torch
from lsp_model import GPT2Tokenizer
from tqdm import tqdm

from env import END_OF_TEXT_TOKEN, TOKEN_TYPE_CXT, TOKEN_TYPE_DOC
from gpt2_training.train_utils import InputFeatures_train as InputFeatures

def _get_file_len(corpus):
    n_line = int(sp.check_output(f"wc -l {corpus}".split(),
                                 universal_newlines=True).split()[0])
    return n_line

def _norm_text(text):
    w, *toks = text.strip().split()
    try:
        w = float(w)
    except Exception:
        toks = [w] + toks
        w = 1.0
    return w, ' '.join(toks)

def _get_inputs_from_text(text, tokenizer):
    doc, srcs, tgt = text.strip().split('\t')
    doc_inputs = tokenizer.encode(doc.strip())

    weights = []
    inputs = []
    for src in srcs.split(' EOS '):

        src_weight = 0.0
        context_id = tokenizer.encode(src.lstrip().rstrip())
        weights.append(src_weight)
        inputs.append(context_id)

    tgt_weight = 1.0
    response_id = tokenizer.encode(tgt)
    weights.append(tgt_weight)
    inputs.append(response_id)

    return doc_inputs, weights, inputs

def _make_features(id_, doc_inputs, weights, inputs, tokenizer, max_len, reverse=False, doc_limit=256, cxt_limit=128, rsp_limit=128 ):
    end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]
    features = []

    doc = doc_inputs[:doc_limit] if len(doc_inputs)>doc_limit else doc_inputs
    cxt = inputs[0]
    for x in inputs[1:-1]:
        cxt += [end_of_text_id] + x
    cxt = cxt[-cxt_limit:] if len(cxt)>cxt_limit else cxt
    rsp = inputs[-1][:rsp_limit] if len(inputs[1])>rsp_limit else inputs[-1]

    sents = [doc, cxt, rsp]
    ws = [0.0, 0.0, 1,0]   
    feat = _make_feature(id_, sents, ws, end_of_text_id, reverse=reverse)

    if feat is not None:
        features.append(feat)

    return features

def _make_feature(id_, sents, ws, eos, reverse=False, doc_start_pos=400):
    if all(w == 0 for w in ws[1:]):
        return None
    lm_labels = []
    weights = []
    token_type_ids = []  
    if not reverse:
        input_ids = [i for s in sents for i in s+[eos]][:-1]
        for i, (s, w) in enumerate(zip(sents, ws)):
            if i == 0:  
                lm_labels += [-1] * len(s)
                weights += [0.0] * len(s)
                token_type_ids += [TOKEN_TYPE_DOC] * (len(s) + 1)
                continue

            token_type_ids += [TOKEN_TYPE_CXT] * (len(s) + 1)
            if w == 0.0:
                lm_labels += [-1] * (len(s) + 1)
                weights += [0.0] * (len(s) + 1)
            else:
                lm_labels += (s + [eos])
                weights += [w] * (len(s) + 1)
    else:
        sents = [sents[2],sents[0], sents[1]] 
        ws = [0.0, 1.0, 1.0]
        input_ids = [i for s in sents for i in s+[eos]][:-1]
        for i, (s, w) in enumerate(zip(sents, ws)):
            if i == 0:  
                lm_labels += [-1] * len(s)
                weights += [0.0] * len(s)
                token_type_ids += [TOKEN_TYPE_CXT] * (len(s) + 1)
                continue
            else:  
                token_type_ids += [TOKEN_TYPE_DOC if i == 1 else TOKEN_TYPE_CXT] * (len(s) + 1)
                lm_labels += (s + [eos])
                weights += [w] * (len(s) + 1)

    token_type_ids = token_type_ids[:-1]

    i = len(lm_labels) - 1
    while i >= 0:
        if lm_labels[i] != -1:
            break
        i -= 1
    input_ids = input_ids[:i+1]
    lm_labels = lm_labels[:i+1]
    weights = weights[:i+1]
    token_type_ids = token_type_ids[:i+1]

    while len(input_ids) % 8 != 0:
        input_ids.append(0)
        token_type_ids.append(TOKEN_TYPE_CXT)
        lm_labels.append(-1)
        weights.append(0.0)

    if not reverse:
        doc_len = len(sents[0])+1
        position_ids = list(range(doc_start_pos, doc_start_pos + doc_len,1)) + list(range(len(input_ids) - doc_len))
    else:
        rsp_len = len(sents[0])+1
        doc_len = len(sents[1])+1
        position_ids = list(range(0, rsp_len,1)) + list(range(doc_start_pos, doc_start_pos + doc_len,1)) + list(range(rsp_len, len(input_ids) - doc_len, 1))

    assert (len(input_ids) == len(position_ids) == len(token_type_ids) == len(lm_labels) == len(weights))
    assert len(input_ids) % 8 == 0
    if len(input_ids) == 0:
        import pdb
        pdb.set_trace()
    feature = InputFeatures(id_, input_ids, position_ids, token_type_ids, lm_labels, weights)
    return feature

def main(args):
    toker = GPT2Tokenizer.from_pretrained('gpt2')
    attrs = []
    if args.reverse:
        attrs.append('reverse')
    if args.two_turn:
        attrs.append('2turn')
    if attrs:
        db_path = (f'{args.corpus[:-4]}.{args.max_seq_len}len.'
                   f'{".".join(attrs)}.db/db')
    else:
        db_path = f'{args.corpus[:-4]}.{args.max_seq_len}len.db/db'
    if exists(dirname(db_path)):
        raise ValueError(f'Found existing DB {dirname(db_path)}, please backup')
    else:
        os.makedirs(dirname(db_path))
    with open(args.corpus, "r", encoding="utf-8") as reader, \
            shelve.open(db_path, 'n') as db:
        chunk = []
        n_chunk = 0
        n_example = 0
        for line in tqdm(reader, total=_get_file_len(args.corpus)):
            try:
                if len(chunk) >= args.chunk_size:

                    db[f'chunk_{n_chunk}'] = gzip.compress(
                        json.dumps(chunk[:args.chunk_size]).encode('utf-8'))
                    chunk = chunk[args.chunk_size:]
                    n_chunk += 1
                if len(line)<25: 
                    continue

                doc_inputs, weights, inputs = _get_inputs_from_text(line, toker)  
                if len(doc_inputs)<args.doc_limit:
                    continue

                if args.two_turn:
                    weights = weights[:2]
                    inputs = inputs[:2]
                if len(weights) < 2:
                    continue

                features = _make_features(n_example, doc_inputs, weights, inputs,
                                            toker, args.max_seq_len, reverse = args.reverse)
                for feature in features:
                    chunk.append(vars(feature))
                    n_example += 1

            except Exception as e:   
                print('!!! prepro exception !!!', e)
                continue

        db[f'chunk_{n_chunk}'] = gzip.compress(
            json.dumps(chunk).encode('utf-8'))

    meta = {'n_example': n_example,
            'chunk_size': args.chunk_size,
            'max_seq_len': args.max_seq_len,
            'reverse': args.reverse,
            'two_turn': args.two_turn}
    with open(join(dirname(db_path), 'meta.json'), 'w') as writer:
        json.dump(meta, writer, indent=4)
    torch.save(toker, join(dirname(db_path), 'tokenizer.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True,
                        help='file name of training corpus (should be .tsv)')
    parser.add_argument('--chunk_size', type=int, default=8192,
                        help='num of data examples in a storing chunk')
    parser.add_argument('--max_seq_len', type=int, default=512,   
                        help='discard data longer than this')
    parser.add_argument('--doc_limit', type=int, default=20,   
                        help='discard data with doc length smaller than this')
    parser.add_argument('--reverse', action='store_true',
                        help='reverse the src tgt')
    parser.add_argument('--two_turn', action='store_true',
                        help='take only the first 2 turns')

    args = parser.parse_args()

    main(args)
