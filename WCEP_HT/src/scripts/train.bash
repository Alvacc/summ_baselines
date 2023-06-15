#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=220108_hiersumm_multinew_parsingsum2_step10w_b13000

CUDA_VISIBLE_DEVICES=$1 python train_abstractive.py \
-mode train \
-batch_size 13000 \
-seed 666 \
-train_steps 100000 \
-save_checkpoint_steps 10000 \
-report_every 100 \
-trunc_tgt_ntoken 400 \
-trunc_src_nblock 24 \
-visible_gpus $1 \
-gpu_ranks 0 \
-world_size 1 \
-accum_count 4 \
-dec_dropout 0.1 \
-enc_dropout 0.1 \
-label_smoothing 0.1 \
-accum_count 4 \
-inter_layers 6,7 \
-inter_heads 8 \
-hier \
-data_path ../../data/multinews/MULTINEWS \
-model_path ../models/${name} > logs/${time}_train_${name}.log 2>&1 &
