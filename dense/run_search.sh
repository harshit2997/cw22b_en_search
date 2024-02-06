sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=80G --time=0 --nodelist=boston-2-25 \
  --wrap="python -m pyserini.search.faiss   --index /ssd/hmehrotr/CW22_ind_en00_00_11_contrmarco_mean   --encoder-class contriever --encoder facebook/contriever-msmarco   --topics /bos/tmp2/hmehrotr/query_3k.tsv   --threads 16   --batch-size 128   --hits 4000   --device cpu   --output /bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_t512_d768_00_11.tsv "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=80G --time=0 --nodelist=boston-2-27 \
  --wrap="python -m pyserini.search.faiss   --index /ssd/hmehrotr/CW22_ind_en00_12_23_contrmarco_mean   --encoder-class contriever --encoder facebook/contriever-msmarco   --topics /bos/tmp2/hmehrotr/query_3k.tsv   --threads 16   --batch-size 128   --hits 4000   --device cpu   --output /bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_t512_d768_12_23.tsv "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=80G --time=0 --nodelist=boston-2-31 \
  --wrap="python -m pyserini.search.faiss   --index /ssd/hmehrotr/CW22_ind_en00_24_35_contrmarco_mean   --encoder-class contriever --encoder facebook/contriever-msmarco   --topics /bos/tmp2/hmehrotr/query_3k.tsv   --threads 16   --batch-size 128   --hits 4000   --device cpu   --output /bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_t512_d768_24_35.tsv "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=80G --time=0 --nodelist=boston-2-29 \
  --wrap="python -m pyserini.search.faiss   --index /ssd/hmehrotr/CW22_ind_en00_36_46_contrmarco_mean   --encoder-class contriever --encoder facebook/contriever-msmarco   --topics /bos/tmp2/hmehrotr/query_3k.tsv   --threads 16   --batch-size 128   --hits 4000   --device cpu   --output /bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_t512_d768_36_46.tsv "

