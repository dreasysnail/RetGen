#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import argparse
import os
import logging
import pathlib
import tqdm
import re
import sys
sys.path.append("./dialogpt")   
from dpr.options import add_encoder_params, add_cuda_params, setup_args_gpu, set_seed, print_args
from generate_dense_embeddings import main as dense_encoding
from generate_dense_embeddings import gen_ctx_vectors
from dense_retriever import main as dense_retrieve
from train_dense_encoder_modified import BiEncoderTrainer
from dense_retriever import *
from dialogpt.gpt2_training.train_utils import boolean_string
from dpr.utils.model_utils import load_states_from_checkpoint_only_model

EVAL_ON_EACH = True


def init_retriever_single_rank(args, eval_on_each = EVAL_ON_EACH):
    # evaluate based on ranking setting
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    print_args(args)
    args.do_lower_case = True
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)

    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    encoder = encoder.question_model
    encoder.to(device)

    encoder.eval()

    args.encoded_ctx_file = args.out_file + '_*'
    if eval_on_each:
        args.encoded_ctx_file = args.out_file + '_' + str(args.shard_id)

    # index all passages
    ctx_files_pattern = args.encoded_ctx_file
    logger.info('encoded files: %s', ctx_files_pattern )
    input_paths = sorted(glob.glob(ctx_files_pattern))
    logger.info('Reading all passages data from files: %s', input_paths)

    # index all passages
    args.encoded_file_patterns = ctx_files_pattern

    # load weights from the model file
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
        index_buffer_sz = -1  # encode all at once
    else:
        index = DenseFlatIndexer(vector_size)

    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)

    args.out_file = os.path.join(os.path.dirname(args.model_file), 'dense_embedding', os.path.basename(args.model_file) + '.' + os.path.basename(args.ctx_file))  
    if hasattr(args, 'load_old_model') and args.load_old_model:
        args.out_file = os.path.join(os.path.dirname(args.model_file), 'dense_embedding', os.path.basename(args.ctx_file))
    if args.encoding:
        dense_encoding(args)
        exit()

    args.encoded_ctx_file = args.out_file + '_*'  
    if eval_on_each:
        args.encoded_ctx_file = args.out_file + '_' + str(args.shard_id)

    # index all passages
    ctx_files_pattern = args.encoded_ctx_file
    logger.info('encoded files: %s', ctx_files_pattern )
    input_paths = sorted(glob.glob(ctx_files_pattern))
    logger.info('Reading all passages data from files: %s', input_paths)

    # index all passages
    args.encoded_file_patterns = ctx_files_pattern


    if args.hnsw_index:
        indexer_file = os.path.join(os.path.dirname(os.path.dirname(ctx_files_pattern)), os.path.basename(args.model_file) + '.' + os.path.basename(args.ctx_file)+'.indexer.cp')
        indexer_file_dpr = os.path.join(os.path.dirname(os.path.dirname(ctx_files_pattern)), os.path.basename(args.model_file) + '.' + os.path.basename(args.ctx_file)+'.indexer.cp.index.dpr')
        if hasattr(args, 'load_old_model') and args.load_old_model:
            indexer_file = os.path.join(os.path.dirname(os.path.dirname(ctx_files_pattern)), os.path.basename(args.ctx_file)+'.indexer.cp')
            indexer_file_dpr = os.path.join(os.path.dirname(os.path.dirname(ctx_files_pattern)), os.path.basename(args.ctx_file)+'.indexer.cp.index.dpr')

        if not os.path.exists(indexer_file_dpr):
            assert ctx_files_pattern is not None, 'encode file patterns cannot be None'
            start_time = time.time()
            input_paths = glob.glob(ctx_files_pattern)
            logger.info('Reading all passages data from files: \n%s', '\n'.join(input_paths))
            logger.info(f'Indexing to file {indexer_file}')
            retriever.index_encoded_data(input_paths, buffer_size=index_buffer_sz)
            print('time cost =', time.time() - start_time, 's')
            retriever.index.serialize(indexer_file)
        else:
            index = DenseHNSWFlatIndexer(vector_size)
            retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)
            retriever.index.deserialize_from(indexer_file)
    else:
        retriever.index_encoded_data(input_paths, buffer_size=index_buffer_sz, remove_duplicates=False)
    
    return retriever


def init_retriever(args, eval_on_each = EVAL_ON_EACH, encoder=None, tensorizer=None, force_index=False, file_suffix = ''):
    args.do_lower_case = True
    # evaluate based on ranking setting
    if encoder is None:
        if not hasattr(args,'load_trained_model') or not args.load_trained_model:
            saved_state = load_states_from_checkpoint(args.model_file)
        else: 
            saved_state = load_states_from_checkpoint_only_model(args.model_file)

        set_encoder_params_from_state(saved_state.encoder_params, args)

        tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

        encoder = encoder.question_model

        # load weights from the model file
        model_to_load = get_model_obj(encoder)
        logger.info('Loading saved model state ...')

        prefix_len = len('question_model.')
        question_encoder_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                                    key.startswith('question_model.')}
        model_to_load.load_state_dict(question_encoder_state)

    encoder.eval()
    vector_size = encoder.get_out_size()
    logger.info('Encoder vector_size=%d', vector_size)
    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                                args.local_rank,
                                                args.fp16)

    index_buffer_sz = args.index_buffer
    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size)
        index_buffer_sz = -1  # encode all at once
    else:
        index = DenseFlatIndexer(vector_size)

    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)
    # TODO

    
    args.out_file = os.path.join(os.path.dirname(args.model_file), 'dense_embedding', file_suffix, os.path.basename(args.model_file) + '.' + os.path.basename(args.ctx_file))  
    if hasattr(args, 'load_old_model') and args.load_old_model:
        args.out_file = os.path.join(os.path.dirname(args.model_file), 'dense_embedding', file_suffix, os.path.basename(args.ctx_file))

    if force_index:
        rows = []
        with open(args.ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            # file format: doc_id, doc_text, title
            ### TODO: potential error
            rows.extend([(row[0], row[1], row[2]) for row in reader if
                         row[0] != 'id' and all([len(row[0]) != 0, len(row[1]) >= 10, len(row[2]) != 0])])

        if hasattr(args, 'change_id_over_card') and args.change_id_over_card:
            args.shard_id = args.local_rank if args.local_rank != -1 else 0


        shard_size = int(len(rows) / args.num_shards)
        start_idx = args.shard_id * shard_size
        end_idx = start_idx + shard_size
        logger.info('Producing encodings to file(s): %s', args.out_file)
        logger.info('Producing encodings for passages range: %d to %d (out of total %d)', start_idx, end_idx, len(rows))
        rows = rows[start_idx:end_idx]

        data = gen_ctx_vectors(rows, encoder, tensorizer, args, False)


        file = args.out_file + '_' + str(args.shard_id)
        pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
        logger.info(' %s' % file)
        with open(file, mode='wb') as f:
            pickle.dump(data, f)

        if args.local_rank != -1:
            torch.distributed.barrier()
    args.encoded_ctx_file = args.out_file + '_*'
    if eval_on_each:
        args.encoded_ctx_file = args.out_file + '_' + str(args.shard_id)


    if args.encoding:
        dense_encoding(args)

    # index all passages
    ctx_files_pattern = args.encoded_ctx_file
    logger.info('encoded files: %s', ctx_files_pattern )
    input_paths = sorted(glob.glob(ctx_files_pattern))
    logger.info('Reading all passages data from files: %s', input_paths)
    # index all passages
    if args.hnsw_index:
        indexer_file = os.path.join(os.path.dirname(os.path.dirname(ctx_files_pattern)), file_suffix, os.path.basename(args.model_file) + '.' + os.path.basename(args.ctx_file)+'.indexer.cp')
        indexer_file_dpr = os.path.join(os.path.dirname(os.path.dirname(ctx_files_pattern)), file_suffix, os.path.basename(args.model_file) + '.' + os.path.basename(args.ctx_file) +'.indexer.cp.index.dpr')
        if hasattr(args, 'load_old_model') and args.load_old_model:
            indexer_file = os.path.join(os.path.dirname(os.path.dirname(ctx_files_pattern)), file_suffix, os.path.basename(args.ctx_file)+'.indexer.cp')
            indexer_file_dpr = os.path.join(os.path.dirname(os.path.dirname(ctx_files_pattern)), file_suffix, os.path.basename(args.ctx_file)+'.indexer.cp.index.dpr')


        if not os.path.exists(indexer_file_dpr) or force_index:
            if args.local_rank not in [-1, 0]:
                torch.distributed.barrier()

            if args.local_rank in [0, -1]:
                assert ctx_files_pattern is not None, 'encode file patterns cannot be None'
                start_time = time.time()
                input_paths = glob.glob(ctx_files_pattern)
                assert len(input_paths) > 0, f"input path {input_paths} needs to be larger than 0"
                logger.info('HNSW: Reading all passages data from files: \n%s', '\n'.join(input_paths))
                logger.info(f'Indexing to file {indexer_file}')
                retriever.index_encoded_data(input_paths, buffer_size=index_buffer_sz)
                print('time cost =', time.time() - start_time, 's, local_rank =', args.local_rank)
                retriever.index.serialize(indexer_file)

            if args.local_rank == 0:
                torch.distributed.barrier()

            if args.local_rank not in [-1, 0]:
                logger.info('file exist and not force index, read from files')
                index = DenseHNSWFlatIndexer(vector_size)
                retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)
                retriever.index.deserialize_from(indexer_file)
        else:
            logger.info('file exist and not force index, read from files')
            index = DenseHNSWFlatIndexer(vector_size)
            retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)
            retriever.index.deserialize_from(indexer_file)

    else:
        retriever.index_encoded_data(input_paths, buffer_size=index_buffer_sz, remove_duplicates=False)

    all_passages = load_passages(args.ctx_file)  # {'1':doc, ctx}
    if len(all_passages) == 0:
        raise RuntimeError('No passages data found. Please specify ctx_file param properly.')
    return retriever, all_passages


def main(args):
    if args.retriever_master_rank and args.local_rank != -1:
        set_seed(args)
        if args.local_rank in [-1, 0]: # only exist in master rank
            retriever = init_retriever_single_rank(args, eval_on_each=args.eval_on_each)
        all_passages = load_passages(args.ctx_file) # exist on all ranks because of ret_passages
        if len(all_passages) == 0:
            raise RuntimeError('No passages data found. Please specify ctx_file param properly.')
    else:
        retriever, all_passages = init_retriever(args, eval_on_each=args.eval_on_each)


    questions = []
    question_answers = []

    for ds_item in parse_qa_csv_file(args.qa_file, simple_parser = True):
        question, answers = ds_item
        questions.append(question)
        question_answers.append(answers)
    questions_tensor = retriever.generate_question_vectors(questions)
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs, is_hnsw = args.hnsw_index) 
    print(top_ids_and_scores[0])

    if not args.save_to:
        # TODO
        save_to = os.path.join(os.path.dirname(args.qa_file),(args.output_name+"." if args.output_name else "") + "top_doc.id.txt")
        if EVAL_ON_EACH:
            save_to = os.path.join(os.path.dirname(args.qa_file), (args.output_name+"." if args.output_name else "") + f"shard{args.shard_id}.top_doc.id.txt")
    else:
        save_to = args.save_to


    if args.output_doc:
        save_to_doc = save_to[:-6] + "doc.txt"
        logger.info('Save top doc to: %s', save_to_doc)
        with open(save_to_doc,'w') as out_d_f:
            with tqdm.tqdm(total=len(top_ids_and_scores), desc=f"Extract top {args.n_docs} docs") as pbar:
                for idx, res in enumerate(tqdm.tqdm(top_ids_and_scores, desc="Iteration")):
                    assert(args.n_docs <= len(res[0]))
                    if args.remove_positive:
                        out_d_f.write("\t".join([all_passages[str(x).strip()][0] for x in res[0][:args.n_print_docs] if all_passages[str(x).strip()][0]!=all_passages[str(idx+1)][0]])+'\n')
                    else:
                        out_d_f.write("\t".join([re.sub('[\t\n]','',all_passages[str(x).strip()][0]) for x in res[0][:args.n_print_docs]])+'\n')


    if args.output_prob:
        save_to_prob = save_to[:-6] + "prob.txt"
        logger.info('Save prob to: %s', save_to_prob)
        with open(save_to_prob,'w') as out_d_p:
            with tqdm.tqdm(total=len(top_ids_and_scores), desc=f"Extract top {args.n_docs} docs") as pbar:
                for idx, res in enumerate(tqdm.tqdm(top_ids_and_scores, desc="Iteration")):
                    assert(args.n_docs <= len(res[0]))
                    if args.remove_positive:
                        out_d_p.write("\t".join([f"{s:.2f}" for x,s in zip(res[0][:args.n_print_docs],res[1][:args.n_print_docs]) if all_passages[str(x).strip()][0]!=all_passages[str(idx+1)][0]])+'\n')
                    else:
                        out_d_p.write("\t".join([f"{s:.2f}" for s in res[1][:args.n_print_docs]]) +'\n')


    logger.info('Save top id to: %s', save_to)
    with open(save_to,'w') as out_f:
        with tqdm.tqdm(total=len(top_ids_and_scores), desc=f"Extract top {args.n_docs} docs") as pbar:
            for idx, res in enumerate(tqdm.tqdm(top_ids_and_scores, desc="Iteration")):
                assert(args.n_docs <= len(res[0]))
                if args.remove_positive:
                    out_f.write("\t".join([str(x) for x in res[0][:args.n_docs] if all_passages[str(x).strip()][0]!=all_passages[str(idx+1)][0]])+'\n')
                else:
                    out_f.write("\t".join([str(x) for x in res[0][:args.n_docs]])+'\n')


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    console = logging.StreamHandler()
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    add_cuda_params(parser)
    add_encoder_params(parser)

    parser.add_argument('--ctx_file', type=str, default=None, help='Path to passages set .tsv file for encoding')
    parser.add_argument('--qa_file', required=True, type=str, default=None,
                        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--save_to', required=False, type=str, default=None,
                        help='output file path to write results to ')
    parser.add_argument('--out_file', required=False, type=str, default=None,
                        help='doc embedding save to')
    parser.add_argument('--seed', type=int, default=42, help="seed for random number generator")
    parser.add_argument('--shard_id', type=int, default=0, help="Number(0-based) of data shard to process")
    parser.add_argument('--num_shards', type=int, default=1, help="Total amount of data shards")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
    parser.add_argument('--n_docs', type=int, default=5, help="Amount of top docs to return")
    parser.add_argument('--n_print_docs', type=int, default=5, help="Amount of top docs to return")
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--index_buffer', type=int, default=50000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
    parser.add_argument("--output_doc", action='store_true', help='If enabled, output doc and id')
    parser.add_argument('--output_name', required=False, type=str, default=None,
                        help='top doc file save to')
    parser.add_argument("--encoding", action='store_true', help='If enabled, encode the doc features offline')
    parser.add_argument("--shard_folder", action='store_true', help='If enabled, encode the doc features are sharded')
    parser.add_argument("--log_batch_step", default=100, type=int, help="")
    parser.add_argument("--remove_positive", action='store_true', help='If enabled, remove positive document from the extraction')
    parser.add_argument("--output_prob", action='store_true', help='If enabled, print doc probs')
    parser.add_argument("--eval_on_each", action='store_true', help='If enabled, eval on each shard')
    parser.add_argument("--retriever_master_rank", type=boolean_string, default=True, help='not for single card')

    
    args = parser.parse_args()
    args.do_lower_case = True
    if args.local_rank == -1:
        logger.info('CUDA available? {}'.format(str(torch.cuda.is_available())))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        args.device, args.n_gpu = device, n_gpu
    else:
        # distributed training
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = torch.distributed.get_world_size()
        args.device, args.n_gpu = device, 1
        logger.info("device: {} n_gpu: {}, distributed training: {}, "
                    "16-bits training: {}".format(
                        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'



    main(args)





