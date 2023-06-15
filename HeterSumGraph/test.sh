#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=test_20220119_HeterSumGraph_baseline_m9

python evaluation.py \
--cuda \
--gpu=$1 \
--data_dir=../data/multinews_HeterSumGraph/datasets/multinews \
--cache_dir=../data/multinews_HeterSumGraph/graphfile/cache/MultiNews \
--embedding_path=../data/multinews_HeterSumGraph/Glove/glove.42B.300d.txt \
--model=HDSG \
--save_root=models/$name \
--log_root=models/$name \
-m=9 \
--test_model=multi \
--use_pyrouge logs/${time}_train_${name}.log 2>&1 &