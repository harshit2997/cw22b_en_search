#!/bin/bash
#SBATCH -n 8 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=10GB # memory
#SBATCH --time=0
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --nodelist=boston-2-29

python bert_ranking.py \
        --model-location /ssd/hmehrotr/models_roberta_marco_no_url \
        --out-path /bos/tmp2/hmehrotr/bert/bert_reranking_roberta_marco_no_url.tsv \
        --fs-hits 1000