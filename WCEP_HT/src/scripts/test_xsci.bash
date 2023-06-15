#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=20220323_multiXScience_basedline_step100w_b5000_min110max150

CUDA_VISIBLE_DEVICES=$1 python train_abstractive.py \
-data_path ../../data/Multi-XScience/texts/newser \
-mode validate  \
-batch_size 5000 \
-valid_batch_size 7500 \
-seed 666 \
-trunc_tgt_ntoken 400 \
-trunc_src_nblock 40 \
-visible_gpus $1 \
-gpu_ranks 0 \
-model_path ../models/220201_hiersumm_org_multiXsci/model_step_1000000.pt \
-inter_layers 6,7 \
-inter_heads 8 \
-hier \
-report_rouge \
-max_wiki 100000  \
-dataset test \
-alpha 0.4 \
-result_path ../summary/${name}/results \
-min_length 110 \
-max_length 150 > logs/${time}_test_${name}.log 2>&1 &

# -log_file ../logs/test_multinews_step1w_20210120.log  \
