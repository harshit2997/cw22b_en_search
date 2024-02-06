#!/bin/bash
#SBATCH -n 8 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=100GB # memory
#SBATCH --time=0

for i in $(seq -f "%03g" 0 183)
do
python helpers/build_train_from_ranking.py \
    --tokenizer_name roberta-base \
    --rank_file /bos/tmp2/hmehrotr/bert/marco_raw_data/bos/tmp11/zhuyund/hdct-marco-train-new/${i}.txt \
    --json_dir /bos/tmp2/hmehrotr/bert/marco_data_no_url \
    --n_sample 10 \
    --sample_from_top 100 \
    --random \
    --truncate 512 \
    --qrel /bos/tmp2/hmehrotr/bert/marco_raw_data/msmarco-doctrain-qrels.tsv.gz \
    --query_collection /bos/tmp2/hmehrotr/bert/marco_raw_data/msmarco-doctrain-queries.tsv \
    --doc_collection /bos/tmp2/hmehrotr/bert/marco_raw_data/msmarco-docs.tsv
done