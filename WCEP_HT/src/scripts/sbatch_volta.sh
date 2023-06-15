#!/bin/bash

# Configure the resources required
#SBATCH -M volta                                                # use volta
#SBATCH -p batch                                                # partition (this is the queue your job will be added to)
#SBATCH -A aiml
#SBATCH -n 1              	                                    # number of tasks (sequential job starts 1 task) (check this if your job unexpectedly uses 2 nodes)
#SBATCH -c 1              	                                    # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=24:00:00                                         # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:1                                            # generic resource required (here requires 4 GPUs)
#SBATCH --mem=8GB                                               # specify memory required per node (here set to 16 GB)

# Configure notifications
##SBATCH --mail-type=END                                         # Type of email notifications will be sent (here set to END, which means an email will be sent when the job is done)
##SBATCH --mail-type=FAIL                                        # Type of email notifications will be sent (here set to FAIL, which means an email will be sent when the job is fail to complete)
##SBATCH --mail-user=a1757791@adelaide.edu.au                    # Email to which notification will be sent

# record GPU utilisation
nvidia-smi -l > logs/nv-smi_sa.log.${SLURM_JOB_ID} 2>&1 &

# Execute your script (due to sequential nature, please select proper compiler as your script corresponds to)
time=$(date "+%Y%m%d-%H%M%S")
name=hiersumm_multinew_addparsing3_step100w_b13000_trainfrom91000_0525

python train_abstractive.py \
-mode train \
-batch_size 13000 \
-seed 666 \
-train_steps 1000000 \
-save_checkpoint_steps 1000 \
-report_every 100 \
-trunc_tgt_ntoken 400 \
-trunc_src_nblock 24 \
-visible_gpus 0 \
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
-train_from ../models/hiersumm_multinew_addparsing3_step100w_b13000_trainfrom88000_0517/model_step_91000.pt \
-data_path ../../data/multinews/MULTINEWS \
-model_path ../models/${name}


# -train_from ../models/hiersumm_multinew_baseline_step100w_b13000_trainfrom70000_0506/model_step_15000.pt \