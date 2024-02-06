while getopts e:p:l:o:s: flag
do
    case "${flag}" in
        e) encoder=${OPTARG};;
        p) pooling=${OPTARG};;
        l) length=${OPTARG};;
        o) output=${OPTARG};;
        s) suffix=${OPTARG};;

    esac
done

last=${output:0-1}

if [ $last != "/" ]; then
    output=${output}/
fi


sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-25 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0000  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_00_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-25 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0001  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_01_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-25 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0002  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_02_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-25 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0003  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_03_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "
sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-25 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0004  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_04_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-25 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0005  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_05_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-25 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en006  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_06_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-25 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0007  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_07_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "
sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-25 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0008  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_08_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-25 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0009  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_09_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-25 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0010  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_10_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-25 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0011  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_11_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "
sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-27 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0012  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_12_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-27 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0013  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_13_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-27 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0014  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_14_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \'
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-27 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0015  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_15_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "
sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-27 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0016  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_16_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-27 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0017  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_17_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-27 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0018  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_18_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-27 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0019  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_19_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "
sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-27 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0020  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_20_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-27 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0021  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_21_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-27 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0022  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_22_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-27 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0023  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_23_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "
sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-31 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0024  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_24_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-31 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0025  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_25_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-31 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0026  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_26_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-31 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0027  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_27_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-31 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0028  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_28_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-31 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0029  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_29_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-31 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0030  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_30_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-31 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0031  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_31_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-31 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0032  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_32_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-31 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0033  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_33_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-31 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0034  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_34_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-31 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0035  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_35_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-29 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0036  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_36_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-29 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0037  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_37_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-29 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0038  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_38_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-29 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0039  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_39_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-29 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0040  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_40_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-29 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0041  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_41_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-29 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0042  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_42_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-29 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0043  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_43_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-29 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0044  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_44_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-29 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0045  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_45_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=45G --time=0 --nodelist=boston-2-29 \
  --wrap="python -m encoder \
  input   --corpus /bos/data0/ClueWeb22_B/txt/en/en00/en0046  \
          --fields Clean-Text \
          --docid-field ClueWeb22-ID \
  output  --embeddings ${output}CW22_ind_en00_46_${suffix}_${pooling} \
          --to-faiss \
  encoder --encoder ${encoder} \
          --pooling ${pooling} \
          --fields Clean-Text \
          --batch 16 \
          --max-length ${length} \
          --device cuda:0 "


