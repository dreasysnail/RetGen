#!/bin/bash

P=models/762M
MODEL=models/dialogpt/large_ft.pkl
ARGS="--no_token_id"

ROOT=tests/test_sampling_small
DEFAULT_ARGS="--generation_length 20 --fp16 False --use_gpu --top_k 10000 --is_sampling sample --seed 42"

exec() {
    PR=$1
    CUDA_VISIBLE_DEVICES=0 python run_gpt2.py $DEFAULT_ARGS $ARGS --top_p $PR --model_name_or_path $P --load_checkpoint $MODEL --test_file $ROOT.in --output_ref --output_file .nucleus.out > $0.n$PR.log 2> $0.n$PR.err
    mv $ROOT.nucleus.out.resp.txt $0.n$PR.out
}

exec 0.1
exec 0.5
exec 0.9
exec 0.95
exec 0.99
