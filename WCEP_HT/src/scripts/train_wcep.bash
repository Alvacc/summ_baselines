#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=20220415_hiersumm_org_multiXsci

CUDA_VISIBLE_DEVICES=$1 python train_abstractive.py \
-mode train \
-batch_size 4500 \
-seed 666 \
-train_steps 1000000 \
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
-data_path ../../data/WCEP/input/newser \
-vocab_path ../../data/spm9998_3.model \
-model_path ../models/${name} > logs/${time}_train_${name}.log 2>&1 &
