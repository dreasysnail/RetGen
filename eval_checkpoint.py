#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import argparse
import os
import logging

from dpr.options import add_encoder_params, add_cuda_params, setup_args_gpu, set_seed, print_args
from generate_dense_embeddings import main as dense_encoding
from dense_retriever import main as dense_retrieve
from train_dense_encoder_modified import BiEncoderTrainer
from dense_retriever import *
from dpr.utils.model_utils import load_states_from_checkpoint_only_model

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

def recall_k(result_ctx_ids: List[Tuple[List[object], List[float]]], topK : int) -> float:

    top_k_hits = 0
    total_ref = len(result_ctx_ids)
    for idx, res in enumerate(result_ctx_ids):
        assert(topK <= len(res[0]))
        candidates = set(res[0][:topK])
        if str(idx+1) in candidates:
            top_k_hits += 1
    recall =  float(top_k_hits/total_ref)
    logger.info(f'Validation results: recall@{topK}:{recall:.3f}')
    return recall

parser = argparse.ArgumentParser()
add_cuda_params(parser)
add_encoder_params(parser)

parser.add_argument('--eval_mode', type=str, default='qa', help="evaluation mode for model")
parser.add_argument('--ctx_file', type=str, default=None, help='Path to passages set .tsv file for encoding')
parser.add_argument('--ict_file', type=str, default=None, help='Path to passages set .tsv file for ict')
parser.add_argument('--qa_file', required=True, type=str, default=None,
                    help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")
parser.add_argument("--hard_negatives", default=1, type=int,
                    help="amount of hard negative ctx per question")
parser.add_argument("--other_negatives", default=1, type=int,
                    help="amount of other negative ctx per question")

parser.add_argument('--out_folder', required=False, type=str, default=None,
                    help='output .tsv file path to write results to ')
parser.add_argument('--seed', type=int, default=42, help="seed for random number generator")

parser.add_argument('--shard_id', type=int, default=0, help="Number(0-based) of data shard to process")
parser.add_argument('--num_shards', type=int, default=1, help="Total amount of data shards")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")

parser.add_argument('--match', type=str, default='string', choices=['regex', 'string'],
                    help="Answer matching logic type")
parser.add_argument('--n_docs', type=int, default=5, help="Amount of top docs to return")
parser.add_argument('--validation_workers', type=int, default=16,
                    help="Number of parallel processes to validate results")
parser.add_argument('--index_buffer', type=int, default=50000,
                    help="Temporal memory data buffer size (in samples) for indexer")
parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
parser.add_argument("--encoding", action='store_true', help='If enabled, encode the doc features offline')
parser.add_argument("--shard_folder", action='store_true', help='If enabled, encode the doc features are sharded')
parser.add_argument("--debug", action='store_true', help='If enabled, debug mode')
parser.add_argument("--load_trained_model", action='store_true', help='If enabled, debug mode')

parser.add_argument("--log_batch_step", default=100, type=int, help="")

args = parser.parse_args()

assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

setup_args_gpu(args)
set_seed(args)
print_args(args)
args.do_lower_case = True

if args.eval_mode == 'qa':

    args.out_file = os.path.join(args.out_folder, 'dense_embedding', os.path.basename(args.ctx_file))
    if args.encoding:
        dense_encoding(args)

    args.encoded_ctx_file = args.out_file + '_*'
    if 'ance' in args.encoder_model_type: 
        args.sequence_length = 64
    args.out_file = os.path.join(args.out_folder, '.'.join(['qa_eval', os.path.basename(args.qa_file)]))
    hit_res = dense_retrieve(args)
elif args.eval_mode == 'ict':

    args.learning_rate, args.adam_eps, args.weight_decay, args.gradient_accumulation_steps = 0.0, 0.0, 0.0, 1
    args.adam_betas = '(0.0, 0.0)'
    args.ict = True
    args.batch_size = 128
    args.dev_file = '../data/DPR/data/wikipedia_split_small/wiki_test.tsv'
    args.dev_batch_size = args.batch_size
    trainer = BiEncoderTrainer(args)
    nll_res = trainer.validate_nll()
elif args.eval_mode == 'rank':

    if not args.load_trained_model:
        saved_state = load_states_from_checkpoint(args.model_file)
    else: 
        saved_state = load_states_from_checkpoint_only_model(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)

    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16)
    encoder.eval()

    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')

    prefix_len = len('question_model.')
    question_encoder_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                              key.startswith('question_model.')}
    model_to_load.load_state_dict(question_encoder_state)
    vector_size = model_to_load.get_out_size()
    logger.info('Encoder vector_size=%d', vector_size)

    index_buffer_sz = args.index_buffer
    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size)
        index_buffer_sz = -1  
    else:
        index = DenseFlatIndexer(vector_size)

    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)

    args.out_file = os.path.join(os.path.dirname(args.model_file), 'dense_embedding', os.path.basename(args.model_file) + '.' + os.path.basename(args.ctx_file))

    if args.shard_folder:
        args.out_file = os.path.join(os.path.dirname(args.model_file), 'dense_embedding/shard', os.path.basename(args.model_file) + '.' + os.path.basename(args.ctx_file))
    if args.encoding:
        dense_encoding(args)

    args.encoded_ctx_file = args.out_file + '_*'

    ctx_files_pattern = args.encoded_ctx_file
    input_paths = glob.glob(ctx_files_pattern)
    logger.info('Reading all passages data from files: %s', input_paths)
    retriever.index_encoded_data(input_paths, buffer_size=index_buffer_sz)

    questions = []
    question_answers = []

    for ds_item in parse_qa_csv_file(args.qa_file):
        question, answers = ds_item
        questions.append(question)
        question_answers.append(answers)

    questions_tensor = retriever.generate_question_vectors(questions)
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs, is_hnsw = args.hnsw_index)
    all_passages = load_passages(args.ctx_file)
    if len(all_passages) == 0:
        raise RuntimeError('No passages data found. Please specify ctx_file param properly.')

    log_file = os.path.join(os.path.dirname(args.model_file),os.path.basename(args.model_file) + '.' + os.path.basename(args.qa_file) + ".log.txt")
    logger.info('Save to: %s', log_file)
    with open(log_file,'w') as log_f:
        for k in range(10, args.n_docs+1, 20):
            recall_at_k = recall_k(top_ids_and_scores, topK=k)
            log_f.write(f'Validation results: recall@{k}:{recall_at_k :.3f}\n')
else:
    raise NotImplementedError()

