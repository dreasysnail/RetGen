#!/bin/bash

P=models/762M
MODEL=models/dialogpt/large_ft.pkl
ARGS="--no_token_id"

ROOT=tests/test_beam_small
DEFAULT_ARGS="--generation_length 20 --top_k 100000 --fp16 False --use_gpu"

exec() {
    CUDA_VISIBLE_DEVICES=0 python run_gpt2.py $DEFAULT_ARGS $ARGS --beam --beam_width $BEAM --model_name_or_path $P --load_checkpoint $MODEL --test_file $ROOT.in --output_ref --output_file .beam$BEAM.out > $0.beam$BEAM.log 2> $0.beam$BEAM.err
    mv $ROOT.beam$BEAM.out.resp.txt $0.beam$BEAM.out
}

BEAM=1
exec
BEAM=20
exec
BEAM=100
exec
