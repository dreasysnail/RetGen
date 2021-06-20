#!/bin/bash

P=models/762M
MODEL=models/dialogpt/large_ft.pkl
ARGS="--no_token_id"

ROOT=tests/test_sampling_small
DEFAULT_ARGS="--generation_length 20 --fp16 False --use_gpu --top_k 10 --is_sampling sample --seed 42"

exec() {
    CUDA_VISIBLE_DEVICES=0 python run_gpt2.py $DEFAULT_ARGS $ARGS --model_name_or_path $P --load_checkpoint $MODEL --test_file $ROOT.in --output_ref --output_file .sampling.out > $0.log 2> $0.err
    mv $ROOT.sampling.out.resp.txt $0.out
}

exec
