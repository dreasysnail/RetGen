#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
'''
* @Date: 2019-04-05 16:50:50
* @author: Yizhe Zhang
* @author: Huggingface (borrowed code)
'''
import copy
import numpy as np
import torch
import torch.nn.functional as F
import pdb


EOS_ID = 50256


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def generate_next_token(model, input_ids, position_ids=None, token_type_ids=None, prev=None, temperature=1, top_k=0, top_p=1.0, sample='greedy', past=None, protected_idx=[]):
    assert sample.lower() in ['sample', 'greedy', 'topk', 'protected_topk'], f'{sample} should be one of [sample, greedy, topk, constrained_topk]'

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    with torch.no_grad():
        if not past:
            hidden_states, past = model.transformer(prev, position_ids, token_type_ids, past=past)
        else:
            hidden_states, past = model.transformer(prev, position_ids, token_type_ids,past=past)
        logits = model.lm_head(hidden_states)
        logits = logits[:, -1, :] / temperature
        if sample.lower() != 'protected_topk' or len(protected_idx) == 0:
            if top_p < 1.0:
                top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            else:
                logits = top_k_logits(logits, k=top_k)
        probs = F.softmax(logits, dim=-1)
        if sample.lower() == 'sample':
            prev = torch.multinomial(probs, num_samples=1)
            return prev, probs[0][prev], past
        elif sample.lower() == 'topk':
            probs_sel, prev = torch.topk(probs, k=top_k, dim=-1)
            return prev, probs_sel, past
        elif sample.lower() == 'protected_topk':
            probs_sel, prev = torch.topk(probs, k=top_k + len(protected_idx), dim=-1)
            missing_idx = np.setdiff1d(protected_idx, prev.cpu())
            if len(missing_idx) > 0:
                # If any protected word is missing, add them back by truncating probs_sel, prev:
                missing_probs = probs[0, missing_idx]
                L = len(probs_sel[0]) - len(missing_probs)
                reorder = (-missing_probs).argsort().cpu()
                missing_probs = missing_probs[reorder]
                missing_idx = missing_idx[reorder]
                probs_sel[0,L:] = missing_probs
                prev[0,L:] = torch.tensor(missing_idx)
            return prev, probs_sel, past
        elif sample.lower() == 'greedy':
            assert top_k == 1, f'just a sanity check, top_k needs to be 1 in greedy mode, instead of {top_k}'
            probs_sel, prev = torch.topk(probs, k=1, dim=-1)
            return prev, probs_sel, past



def generate_sequence(model, input_ids, position_ids=None, token_type_ids=None, start_token=None, temperature=1, top_k=0, top_p=1.0, length=20, sample='greedy', past=None, device='cuda', no_token_id = False):
    output = input_ids.new_zeros([input_ids.size(0),0])
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    prev = input_ids
    for i in range(length):
        prev, probs, past = generate_next_token(model, input_ids, position_ids, token_type_ids, prev, temperature, top_k, top_p, sample, past)
        if not no_token_id:
            position_ids = position_ids[:,-1].view(-1,1) + 1
            token_type_ids = token_type_ids[:,-1].view(-1,1)
        output = torch.cat((output, prev), dim=1)
    return output


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


def cut_seq_to_eos(sentence, remove_id=[-1]):
    sent=[]
    for s in sentence:
        if s in remove_id:
            continue
        if s != EOS_ID:
            sent.append(s)
        else:
            break
    return sent


def torch_vec_to_str(x, tokenizer):
    xx = x.cpu().numpy()
    decode_str = [tokenizer.decode(cut_seq_to_eos(s)).encode('utf-8').decode('utf-8') for s in xx]
    return decode_str
