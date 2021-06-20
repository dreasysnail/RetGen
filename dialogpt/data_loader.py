import gzip
import json
import math
import random
import shelve
import torch

import subprocess as sp

from math import ceil
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence

from env import END_OF_TEXT_TOKEN, TOKEN_TYPE_CXT, TOKEN_TYPE_DOC
from gpt2_training.train_utils import (InputFeatures, InputFeatures_train,
                                       RedditExample)

class BucketSampler(Sampler):
    """
    this sampler will sort data by sequence length
    """
    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, shuffle=True):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=lambda i: self._lens[i], reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        batches = [bucket[i:i+self._batch_size]
                   for bucket in buckets
                   for i in range(0, len(bucket), self._batch_size)]
        if self._droplast:
            batches = [batch for batch in batches
                       if len(batch) == self._batch_size]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size]
                        * (len(self._lens) // self._bucket_size)
                        + [len(self._lens) % self._bucket_size])
        if self._droplast:
            return sum(s//self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)

class GPT2FeatureDataset(Dataset):
    """ pytorch dataset for GPT2 training """
    def __init__(self, features, max_len=None):
        self.features = features
        self.max_len = max_len  

    def __getitem__(self, i):
        feat_dict = self.features[i]
        if self.max_len is not None and feat_dict['input_len'] > self.max_len:

            feat_dict['input_ids'] = feat_dict['input_ids'][-self.max_len:]
            feat_dict['position_ids'] = feat_dict['position_ids'][
                -self.max_len:]
            feat_dict['token_type_ids'] = feat_dict['token_type_ids'][
                -self.max_len:]
            feat_dict['lm_labels'] = feat_dict['lm_labels'][-self.max_len:]
        try:
            for s in ['context_len', 'response_len']:
                if s in feat_dict.keys():
                    print("db file missing "+s)
                    del feat_dict[s]
        except Exception:
            import pdb
            pdb.set_trace()

        feat = InputFeatures_train(**feat_dict)
        return feat

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features):
        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long)
                                  for f in features],
                                 batch_first=True, padding_value=0)
        position_ids = pad_sequence([torch.tensor(f.position_ids,
                                                  dtype=torch.long)
                                     for f in features],
                                    batch_first=True, padding_value=0)
        token_type_ids = pad_sequence([torch.tensor(f.token_type_ids,
                                                    dtype=torch.long)
                                       for f in features],
                                      batch_first=True, padding_value=0)
        labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-1)
        return (input_ids, position_ids, token_type_ids, labels)

class BucketingDataLoader(object):
    """ this loads shelve db chunks and then convert to mini-batch loader"""
    def __init__(self, db_name, batch_size, max_seq_len,
                 bucket=100, shuffle=True):
        self.db = shelve.open(f'{db_name}/db', 'r')
        self.batch_size = batch_size
        self.max_len = max_seq_len
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle

    def _get_keys(self):
        keys = list(self.db.keys())
        return keys

    def __iter__(self):
        keys = self._get_keys()

        if self.shuffle:
            random.shuffle(keys)
        for key in keys:
            chunk = json.loads(gzip.decompress(self.db[key]).decode('utf-8'))

            trunc_chunk = []
            lens = []
            for feat in chunk:

                if feat['input_len'] > self.max_len:
                    print("maximum length exceed!!")
                    continue
                trunc_chunk.append(feat)
                lens.append(feat['input_len'])

            dataset = GPT2FeatureDataset(trunc_chunk, self.max_len)
            sampler = BucketSampler(lens, self.bucket_size, self.batch_size,
                                    droplast=True, shuffle=self.shuffle)
            loader = DataLoader(dataset, batch_sampler=sampler,
                                num_workers=0,  
                                collate_fn=GPT2FeatureDataset.collate)
            yield from loader

    def __len__(self):
        raise NotImplementedError()

    def __del__(self):
        self.db.close()

class DistributedBucketingDataLoader(BucketingDataLoader):
    """ distributed version """
    def __init__(self, rank, num_replica, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank
        self.num_replica = num_replica

    def _get_keys(self):
        keys = list(self.db.keys())[self.rank::self.num_replica]
        return keys

def convert_examples_to_features_dynamic(examples, tokenizer,
                                         max_seq_length=512, doc_limit=256, cxt_limit=128, rsp_limit=128, doc_start_pos=400):
    """
    do not pad
    """
    def featurize(example):
        conv_id = example.conv_id

        end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]

        context_id =  [tokenizer.encode(x) + [end_of_text_id] for x in example.context.split(' EOS ')]
        context_id =  [x for sublist in context_id for x in sublist]
        context_id = context_id[:-1]

        doc_id = tokenizer.encode(example.doc)

        response_id = tokenizer.encode(example.response)

        doc_id = doc_id[:doc_limit] if len(doc_id)>doc_limit else doc_id
        context_id = context_id[-cxt_limit:] if len(context_id)>cxt_limit else context_id
        response_id = response_id[:rsp_limit] if len(response_id)>rsp_limit else response_id

        input_ids = doc_id + [end_of_text_id] + context_id + [end_of_text_id] + response_id  

        lm_labels = [-1] * len(doc_id) + [-1] * len(context_id) + [-1] + response_id + ([end_of_text_id] if len(response_id) >0 else [-1]) 

        doc_len = len(doc_id)+1

        position_ids = list(range(doc_start_pos, doc_start_pos + doc_len,1)) + list(range(len(input_ids) - doc_len))   

        token_type_ids = [TOKEN_TYPE_DOC] * doc_len  +  [TOKEN_TYPE_CXT] * (len(input_ids) - doc_len)

        assert (len(input_ids) == len(position_ids) == len(token_type_ids) == len(lm_labels))

        return InputFeatures(conv_id, input_ids, position_ids, token_type_ids,
                             lm_labels, len(doc_id), len(context_id), len(response_id))

    features = [f for f in [featurize(ex) for ex in examples] if f is not None]
    return features

def convert_examples_to_features_dynamic_rev(examples, tokenizer,
                                         max_seq_length=512, doc_limit=256, cxt_limit=128, rsp_limit=128, doc_start_pos=400):
    """
    do not pad
    """
    def featurize(example):
        conv_id = example.conv_id
        end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]
        context_id =  [tokenizer.encode(x) + [end_of_text_id] for x in example.context.split(' EOS ')]
        context_id =  [x for sublist in context_id for x in sublist]
        context_id = context_id[:-1]
        doc_id = tokenizer.encode(example.doc)

        response_id = tokenizer.encode(example.response)

        doc_id = doc_id[:doc_limit] if len(doc_id)>doc_limit else doc_id
        context_id = context_id[-cxt_limit:] if len(context_id)>cxt_limit else context_id
        response_id = response_id[:rsp_limit] if len(response_id)>rsp_limit else response_id

        input_ids = response_id + [end_of_text_id] + doc_id + [end_of_text_id] + context_id 

        lm_labels = [-1] * len(response_id) + doc_id  + context_id + [end_of_text_id] 

        rsp_len = len(response_id)+1
        doc_len = len(doc_id)+1
        position_ids = list(range(0, rsp_len,1)) + list(range(doc_start_pos, doc_start_pos + doc_len,1)) + list(range(rsp_len, len(input_ids) - doc_len, 1))
        token_type_ids = [TOKEN_TYPE_CXT] * rsp_len  + [TOKEN_TYPE_DOC] * doc_len +  [TOKEN_TYPE_CXT] * (len(input_ids) - doc_len - rsp_len)

        assert (len(input_ids) == len(position_ids) == len(token_type_ids) == len(lm_labels))

        return InputFeatures(conv_id, input_ids, position_ids, token_type_ids,
                             lm_labels, len(doc_id), len(context_id), len(response_id))

    features = [f for f in [featurize(ex) for ex in examples] if f is not None]
    return features

class DynamicBatchingLoader(object):
    """ this loader takes raw text file, used for validate perplexity """
    def __init__(self, corpus_file, tokenizer, normalize_data,
                 batch_size, max_seq_length, reverse=False):
        self.corpus = corpus_file
        self.toker = tokenizer
        self.norm = normalize_data
        self.bs = batch_size
        self.max_seq_length = max_seq_length
        self.num_examples = self.get_len(corpus_file)
        self.reverse = reverse

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples/self.bs)

    def _iter_epoch(self):
        try:
            with open(self.corpus, 'r', encoding="utf-8") as corpus:
                i = 0
                while True:
                    examples = []
                    cur_bs = 0
                    while True:
                        line = next(corpus).encode('utf-8').decode('utf-8')
                        contents = line.split('\t')
                        doc, src, tgt_all = contents[0], contents[1], contents[2:]
                        for tgt in tgt_all:
                            if self.norm:
                                doc_line = ' '.join(doc.strip().split())
                                src_line = ' '.join(src.strip().split())
                                tgt_line = ' '.join(tgt.strip().split())
                            else:
                                doc_line = doc.strip()
                                src_line = src.strip()
                                tgt_line = tgt.strip()
                            examples.append(
                                RedditExample(i, doc_line, src_line, tgt_line),
                            )
                            i += 1
                            cur_bs += 1
                        if cur_bs >= self.bs:
                            break
                    if self.reverse:
                        features = convert_examples_to_features_dynamic_rev(
                            examples, self.toker, self.max_seq_length)
                    else:
                        features = convert_examples_to_features_dynamic(
                            examples, self.toker, self.max_seq_length)
                    batch = self._batch_feature(features)
                    yield batch
        except StopIteration:
            pass

    def _batch_feature(self, features):
        input_ids = pad_sequence([torch.tensor(f.choices_features['input_ids'],
                                               dtype=torch.long)
                                  for f in features],
                                 batch_first=True, padding_value=0)
        position_ids = pad_sequence(
            [torch.tensor(f.choices_features['position_ids'], dtype=torch.long)
             for f in features],
            batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(
            [torch.tensor(f.choices_features['token_type_ids'],
                          dtype=torch.long)
             for f in features],
            batch_first=True, padding_value=TOKEN_TYPE_CXT)
        labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-1)
        doc_len = torch.tensor([f.doc_len for f in features],
                                    dtype=torch.long)
        context_len = torch.tensor([f.context_len for f in features],
                                   dtype=torch.long)
        response_len = torch.tensor([f.response_len for f in features],
                                    dtype=torch.long)

        return (input_ids, position_ids, token_type_ids, labels, doc_len,
                context_len, response_len)

    def get_len(self, corpus):
        n_line = int(sp.check_output(f"wc -l {corpus}".split(),
                                     universal_newlines=True).split()[0])
        return n_line
