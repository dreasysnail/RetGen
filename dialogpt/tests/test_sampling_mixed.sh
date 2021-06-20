#!/bin/bash

P=models/762M
MODEL=models/dialogpt/large_ft.pkl
ARGS="--no_token_id"

DEFAULT_ARGS="--generation_length 25 --fp16 False --use_gpu --top_k 10000 --is_sampling sample --seed 42 --batch_size 128"

exec() {
    PR=$1
    K=$2
    ID=p${1}.k${2}
    CUDA_VISIBLE_DEVICES=0 python run_gpt2.py $DEFAULT_ARGS $ARGS --top_p $PR --top_k $K --model_name_or_path $P --load_checkpoint $MODEL --test_file $ROOT.in --output_ref --output_file .nucleus.out > $ROOT.$ID.log 2> $ROOT.$ID.err
    mv $ROOT.nucleus.out.resp.txt $ROOT.$ID.out
}

ROOT=tests/test_sampling_small
exec 0.95 10
