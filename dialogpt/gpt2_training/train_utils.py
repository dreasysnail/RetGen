
import os
import logging
import torch
from collections import defaultdict

from env import END_OF_TEXT_TOKEN, TOKEN_TYPE_DOC, TOKEN_TYPE_CXT
from lsp_model.optim import warmup_linear, noam_decay, noamwd_decay
import re
import numpy as np

logger = logging.getLogger(__name__)

SEQ_LENGTH_SHRINK_PROP = 0.9

def load_model(model, checkpoint, args, verbose=False, set_type_embedding_to_zero=False):
    n_gpu = args.n_gpu
    device = args.device
    if checkpoint is None or checkpoint == "None":
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading finetuned model from %s' % checkpoint)
        model_state_dict = torch.load(checkpoint, map_location="cpu")
        model_state_dict = fix_state_dict_namespace(model_state_dict)

        if set_type_embedding_to_zero: 
            if args.init_checkpoint.endswith('pkl'):

                tmp = model_state_dict['transformer.wte.weight'].type('torch.FloatTensor')
                tmp = tmp[TOKEN_TYPE_CXT,:]

                model_state_dict['transformer.wte.weight'][TOKEN_TYPE_CXT,:] = tmp * 0
                model_state_dict['transformer.wte.weight'][TOKEN_TYPE_DOC,:] = tmp * 0
                model_state_dict['lm_head.decoder.weight'][TOKEN_TYPE_CXT,:] = tmp * 0
                model_state_dict['lm_head.decoder.weight'][TOKEN_TYPE_DOC,:] = tmp * 0
            else:                   
                tmp = model_state_dict['wte.weight'][TOKEN_TYPE_CXT,:]
                model_state_dict['wte.weight'][TOKEN_TYPE_CXT,:] = tmp * 0
                model_state_dict['wte.weight'][TOKEN_TYPE_DOC,:] = tmp * 0
                model_state_dict['lm_head.decoder.weight'][TOKEN_TYPE_CXT,:] = tmp * 0
                model_state_dict['lm_head.decoder.weight'][TOKEN_TYPE_DOC,:] = tmp * 0
        start_model = model
        if (hasattr(model, "transformer")
            and all(not s.startswith('transformer.')
                    for s in model_state_dict.keys())):
            logger.info('loading transfomer only')
            start_model = model.transformer
        start_model.load_state_dict(model_state_dict, strict=True)

    model.to(device)
    return model

def fix_state_dict_namespace(model_state_dict):
    old_keys = []
    new_keys = []
    for t in model_state_dict:
        new_key = t
        if t.startswith('module.'):
            new_key = t.replace('module.', '')
        old_keys.append(t)
        new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    return model_state_dict

class InputFeatures(object):
    def __init__(self, conv_id, input_ids, position_ids, token_type_ids,
                 lm_labels, doc_len, context_len, response_len):
        self.conv_id = conv_id
        self.choices_features = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids
        }
        self.lm_labels = lm_labels
        self.doc_len = doc_len
        self.context_len = context_len
        self.response_len = response_len    

class InputFeatures_train(object):
    def __init__(self, conv_id, input_ids, position_ids, token_type_ids,
                 lm_labels, weights, input_len=None):
        self.conv_id = conv_id
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.token_type_ids = token_type_ids
        self.lm_labels = lm_labels
        self.weights = weights
        if input_len is None:
            self.input_len = len(input_ids)
        else:
            self.input_len = input_len

class RedditExample(object):
    def __init__(self, conv_id, doc, context, response):
        self.conv_id = conv_id
        self.doc = doc
        self.context = context
        self.response = response

    def __repr__(self):
        return 'conv_id = {}\ndoc = {}\ncontext = {}\nresponse = {}'.format(
            self.conv_id, self.doc, self.context, self.response)

    def __str__(self):
        return self.__repr__()

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def get_eval_list_same_length(input_file, tokenizer, max_batch_size,
                              norm=True, for_eval = False):
    examples = []
    with open(input_file, 'r', encoding="utf-8") as f:
        content = [l.split('\t') for l in f.read().splitlines()]
    docs, context, response = [c[0] for c in content], [c[1] for c in content], [c[2:] for c in content]
    i = 0
    for doc, src, tgt_all in zip(docs, context, response):
        for tgt in tgt_all:
            if norm:
                doc_line = ' '.join(doc.strip().split())
                src_line = ' '.join(src.strip().split())
                tgt_line = ' '.join(tgt.strip().split())
            else:
                doc_line = doc.strip()
                src_line = src.strip()
                tgt_line = tgt.strip()
            examples.append(RedditExample(i, doc_line, src_line, tgt_line))
            i += 1

    def featurize(example, doc_limit=256, cxt_limit=128, rsp_limit=128, doc_start_pos=400):
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

        input_ids = doc_id + [end_of_text_id] + context_id + [end_of_text_id]

        lm_labels = response_id

        doc_len = len(doc_id)+1

        position_ids = list(range(doc_start_pos, doc_start_pos + doc_len,1)) + list(range(len(input_ids) - doc_len))  

        token_type_ids = [TOKEN_TYPE_DOC] * doc_len  +  [TOKEN_TYPE_CXT] * (len(input_ids) - doc_len)

        assert (len(input_ids) == len(position_ids) == len(token_type_ids))
        return InputFeatures(conv_id, input_ids, position_ids, token_type_ids,
                             lm_labels, len(doc_id), len(input_ids), len(response_id))

    def batch_feature_same_len(features):

        input_ids = torch.stack([torch.tensor(f.choices_features['input_ids'],
                                              dtype=torch.long)
                                 for f in features])
        position_ids = torch.stack(
            [torch.tensor(f.choices_features['position_ids'], dtype=torch.long)
             for f in features])
        token_type_ids = torch.stack(
            [torch.tensor(f.choices_features['token_type_ids'],
                          dtype=torch.long)
             for f in features])
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f.lm_labels, dtype=torch.long) for f in features],
            batch_first=True, padding_value=-1)
        doc_len = torch.tensor([f.doc_len for f in features],
                                   dtype=torch.long)
        context_len = torch.tensor([f.context_len for f in features],
                                   dtype=torch.long)
        response_len = torch.tensor([f.response_len for f in features],
                                    dtype=torch.long)
        return (input_ids, position_ids, token_type_ids, labels,
                context_len, response_len)

    features = [featurize(e) for e in examples]
    dataloader_pre = defaultdict(list)
    for f in features:
        dataloader_pre[f.context_len + f.response_len if for_eval else f.context_len].append(f)

    dataloader = []
    for l in sorted(dataloader_pre):
        f = batch_feature_same_len(dataloader_pre[l])
        if len(f[0]) <= max_batch_size:
            dataloader.append(f)
        else:
            start_index = 0
            while True:
                dataloader.append([ff[start_index:start_index + max_batch_size]
                                   for ff in f])
                start_index += max_batch_size
                if start_index >= len(f[0]):
                    break
    return dataloader

def get_eval_list_same_length_with_order(input_file, tokenizer, max_batch_size, norm=True, for_eval=False, sort_data=True):

    examples = []
    with open(input_file, 'r', encoding="utf-8") as f:
        content = [l.split('\t') for l in f.read().splitlines()]

    docs, context, response = [c[0] for c in content], [c[1] for c in content], [c[2:] for c in content]
    i = 0
    for doc, src, tgt_all in zip(docs, context, response):
        if norm:
            doc_line = ' '.join(doc.strip().split())
            src_line = ' '.join(src.strip().split())
            tgt_line = [' '.join(tgt.strip().split()) for tgt in tgt_all]
        else:
            doc_line = doc.strip()
            src_line = src.strip()
            tgt_line = [tgt.strip() for tgt in tgt_all]
        examples.append(RedditExample(i, doc_line, src_line, tgt_line))
        i += 1

    def multi_turn_enc(sent):
        sent = re.split('<\|endoftext\|>|EOS', sent) 
        return list(np.concatenate([tokenizer.encode(s.strip()) + [tokenizer.encoder[END_OF_TEXT_TOKEN]] for s in sent]))

    def featurize(example, doc_limit=256, cxt_limit=128, rsp_limit=128, doc_start_pos=400):  
        conv_id = example.conv_id
        end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]
        context_id =  [tokenizer.encode(x) + [end_of_text_id] for x in example.context.split(' EOS ')]
        context_id =  [x for sublist in context_id for x in sublist]
        context_id = context_id[:-1]
        doc_id = tokenizer.encode(example.doc)

        response_id = multi_turn_enc(example.response[0]) if for_eval else None

        doc_id = doc_id[:doc_limit] if len(doc_id)>doc_limit else doc_id
        context_id = context_id[-cxt_limit:] if len(context_id)>cxt_limit else context_id
        if response_id:
            response_id = response_id[:rsp_limit] if len(response_id)>rsp_limit else response_id

        input_ids = doc_id + [end_of_text_id] + context_id + [end_of_text_id]

        lm_labels = response_ids if for_eval else example.response

        doc_len = len(doc_id)+1

        position_ids = list(range(doc_start_pos, doc_start_pos + doc_len,1)) + list(range(len(input_ids) - doc_len))  

        token_type_id = [TOKEN_TYPE_DOC] * doc_len  +  [TOKEN_TYPE_CXT] * (len(input_ids) - doc_len)
        assert (len(input_ids) == len(position_ids) == len(token_type_id))

        return InputFeatures(conv_id, input_ids, position_ids, token_type_id,
                             lm_labels, len(doc_id), len(input_ids)-1, len(response_ids)-1 if for_eval else -1)

    def batch_feature_same_len(features):
        if for_eval:
            input_ids = [torch.tensor(f.choices_features['input_ids'], dtype=torch.long) for f in features]
            position_ids = [torch.tensor(f.choices_features['position_ids'], dtype=torch.long) for f in features]
            token_type_ids = [torch.tensor(f.choices_features['token_type_ids'], dtype=torch.long) for f in features]
            labels = [torch.tensor(f.lm_labels, dtype=torch.long) for f in features]
        else:
            input_ids = torch.stack([torch.tensor(f.choices_features['input_ids'], dtype=torch.long) for f in features])
            position_ids = torch.stack([torch.tensor(f.choices_features['position_ids'], dtype=torch.long) for f in features])
            token_type_ids = torch.stack([torch.tensor(f.choices_features['token_type_ids'], dtype=torch.long) for f in features])
            labels = [f.lm_labels for f in features]
        doc_len = torch.tensor([f.doc_len for f in features], dtype=torch.long)
        context_len = torch.tensor([f.context_len for f in features], dtype=torch.long)
        response_len = torch.tensor([f.response_len for f in features], dtype=torch.long)
        conv_ids = torch.tensor([torch.tensor(f.conv_id, dtype=torch.long) for f in features])

        return input_ids, position_ids, token_type_ids, labels, doc_len, context_len, response_len, conv_ids

    features = [featurize(e) for e in examples]
    dataloader_pre = defaultdict(list)
    for f in features:
        dataloader_pre[f.context_len + f.response_len if for_eval else f.context_len].append(f)

    dataloader = []
    if sort_data:
        sorted_data = sorted(dataloader_pre)
    else:
        sorted_data = dataloader_pre
    for l in sorted_data:
        f = batch_feature_same_len(dataloader_pre[l])
        if len(f[0]) <= max_batch_size:
            dataloader.append(f)
        else:
            start_index = 0
            while True:
                dataloader.append([ff[start_index:start_index + max_batch_size] for ff in f])
                start_index += max_batch_size
                if start_index >= len(f[0]):
                    break
    return dataloader

def set_lr(optimizer, step, schedule, lr,
           warmup_steps, warmup_proportion, n_embd, tot_steps):
    if schedule == 'None':
        lr_this_step = lr
    elif schedule == 'noam':  
        lr_this_step = lr * 1e4 * noam_decay(step+1, warmup_steps, n_embd)
    elif schedule == 'noamwd':  
        lr_this_step = lr * 1e4 * noamwd_decay(step+1, warmup_steps, n_embd)
    else:
        lr_this_step = lr * warmup_linear(step / tot_steps,
                                          warmup_proportion)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step
