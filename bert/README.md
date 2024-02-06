# Training and using rerankers
The `bert` directory contains code to train rerankers on inlink data and host a service that performs BM25 first stage retrieval followed by reranking. The code uses the [Reranker](https://github.com/luyug/Reranker/tree/main) library. We also cover how to train on MS-MARCO document ranking data.

## Setup
Follow instructions given in the Reranker library repository to clone and install it as follows:
```
git clone https://github.com/luyug/Reranker.git
cd Reranker
pip install .
```

## Creating inlink data
While the inlink data is available on the CMU cluster, it needs to be put into a certain JSON format for the Reranker library to be able to use it for training. The `create_reranking_data.py` script is used for this purpose. There are around 4.5M inlink queries in the dataset. Transforming all this data at once takes time. So, we do it parallelly in partitions. The script takes the following arguments:

- `tokenizer_name`: The HuggingFace/local tokenizer to be used to tokenize text
- `truncate`: Maximum number of tokens in a sequence (default is 512)
- `start`: Which query to start from
- `end`: Which query to end at
- `partition`: Index of the partition. Used in the name of the stored JSON file.

An example command to run this script as a SLURM job:

```
sbatch -n 8 -N 1 --mem=45G --time=0 --nodelist=boston-2-9 --wrap="python create_reranking_data.py --tokenizer_name roberta-base --truncate 512 --start 3000001 --end 3500000 --partition 7"
```

This would prepare data from 500k queries (3M-3.5M) and name the stored file `part_7.json`. The path of the directory where this file is stored is hardcoded in the script. Please modify the `out_file` variable in the file to change that. The script stores the document id, title, and body of the document.

## Preparing MS-MARCO data
We also experimented with training on MS-MARCO document ranking data. Download the raw data and unpack it - HDCT+BM25 ranking, corpus, relevance judgements, etc - as given [here](https://github.com/luyug/Reranker/tree/main/examples/msmarco-doc). After doing this, use the `bert/marco_code/helpers/build_train_from_ranking.py` script to generate JSON files for reranker training. `bert/marco_code/run_build_data.sh` runs this operation as a SLURM job. Navigate to the `bert/marco_code` directory and run `bash run_build_data.sh`. This bash file passes parameters for the name of the tokenizer and paths to various MS-MARCO raw data files to `build_train_from_ranking.py`. Modify these arguments according to your setup. The `sample_from_top` parameter controls how many of the top results to take from a ranking to sample negatives. The `n_sample` parameter controls the number of negatives to be randomly sampled from this top section of the ranking. You can also pass the `url` flag to `build_train_from_ranking.py` to indicate that the URL of the document should also be used along with title and body. 

## Training rerankers
Now that we have the training data in a single or multiple JSON files, we can train a reranking model. The `train.py` script does this and is run as a slurm job by `run_train.sh`. You can run it by calling `bash run_train.sh`. Some important parameters to modify in the bash script according to your need:

- `--nproc_per_node`: Number of GPUs to use for training. Make sure this number is less than or equal to the number of GPUs specified in the `--gpus` parameter of `sbatch`.
- `--output_dir`: Directory to store model checkpoints
- `--model_name_or_path`: Local model path of HuggingFace model to start with
- `--save_steps`: Frequency in iterations of storing checkpoints
- `--train_dir`: Path of directory with training data JSON file(s). Can be from the inlink data or MS-MARCO or any other dataset. 
- `--max_len`: Maximum sequence length. Keep same as the one used while creating data.

There are some other parameters being passed which are hyperparameters for training and self-explanatory from the names.

## Generating rankings
The script `bert_ranking.py` does the entire search operation of first performing first stage retrieval using BM25 and then reranking the results using a reranker. A few details about this script:

1. It uses the 2110 queries given in `query_3k.tsv`.
2. For BM25 results, it relies on the BM25 services that can be hosted as described in the `README` file in the `lucene` directory of this repository.
3. BM25 search API URLs are set in the variable `base_urls`. Modify it according to your setup.
4. The first 4000 characters of the full text are used (line 107) since we think this will be enough to get 512 tokens (the sequence length limit).
5. Batch size of 256 is used (line 116) while reranking results.

The bash script `run_bert_ranking.sh` can be used to to run the reranking operation as a SLURM job. The scrip passes 3 parameters to `bert_ranking.py`:

1.`model-location`: The location of the model. If it is a specific checkpoint then its upper level directory should have the checkpoint. If not, then it should have the model and the tokenizer.
2.`out-path`: Path to store the final ranking in TREC format.
3.`fs-hits`: Number of results to get from first stage retrieval.

## Reranking service
The `bert_flask.py` script hosts the reranking operation described in the previos section as a Flask service given a query and number of first stage retrieval results. The top 10 results are finally returned after reranking. You can specify the batch size, length of full text to be used, and BM25 service URLs in `bert_flask.py` similar to what was described for `bert_ranking.py` in the previous section. To start the service run the following SLURM command:

```
sbatch -n 8 -N 1 --mem=80G --time=0 --nodelist=boston-2-29 --wrap="python bert_flask.py"
```

To test the search API, you can try this command after taking the IP and port from the SLURM log file of the job running the service:

```
curl --request GET "http://10.1.1.38:7001/search/bert?query=Where+can+I+find+the+best+prices+on+a+new+Macbook+Pro+laptop&hits=1000"
```

The `hits` parameter specifies the number of first stage retrieval results (default 1000). Finally, the top 10 results are returned.



