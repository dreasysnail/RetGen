# model training
CUDA_VISIBLE_DEVICES=0 python joint_training.py \
    --model_name_or_path configs\
    --init_checkpoint models/reddit_generator.pkl \
    --train_input_file data/reddit_train.db \
    --eval_input_file data/reddit_test.txt \
    --output_dir outputs/joint_reddit \
    --file_suffix joint_reddit \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 1 \
    --num_optim_steps 16000 \
    --encoder_model_type ance_roberta \
    --pretrained_model_cfg bert-base-uncased \
    --model_file models/reddit_retriever.pkl \
    --ctx_file data/wiki.txt \
    --num_shards 1 \
    --batch_size 128 \
    --n_docs 2 \
    --encoding \
    --load_trained_model      

# evaluating checkpoint  hf_bert
CUDA_VISIBLE_DEVICES=0 python eval_checkpoint.py \
        --eval_mode rank \
        --encoder_model_type ance_roberta \
        --pretrained_model_cfg bert-base-uncased \
        --model_file models/reddit_retriever.pkl \
        --qa_file data/2k_positive.txt \
        --ctx_file data/10k.txt \
        --n_docs 50 \
        --batch_size 64 \
        --shard_id 0 \
        --num_shards 1 \
        --load_trained_model \
        --encoding