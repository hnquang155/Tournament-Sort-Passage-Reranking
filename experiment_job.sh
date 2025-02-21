#!/bin/bash
#SBATCH --job-name=job1
#SBATCH --output=log.%x.job_%j
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --mail-user=s224352548@deakin.edu.au
#SBATCH --mail-type=ALL

# -----------------------------------------------------------------------------------------------
T1=$(date +%s)                          # Record how long all this takes to complete

#place your script here
python  new_experiment.py \
  --model_name "lmsys/vicuna-7b-v1.5" \
  --model_short_name "vicuna" \
  --method_wise "listwise" \
  --scoring "generation" \
  --sort_method "heapsort" \
  --r_tournament 1 \
  --shuffle_ranking "original" \
  --parent_dataset "beir" \
  --dataset "trec-covid" \
  --retrieve_step "bm25-flat" \
  --hits 100 \
  --query_length 32 \
  --passage_length 100 \
  --num_child 4


# #########################################################################################
# How long did this take?
T2=$(date +%s)
ELAPSED_TIME=$((T2 - T1))
echo >> env_info.txt
echo "Script has taken $ELAPSED_TIME second(s) to complete" >> env_info.txt