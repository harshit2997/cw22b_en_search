# Creating and searching on dense indexes

The `dense` directory contains all relevant code to create and search on dense indexes. The code creates Faiss "Flat" indexes which support exhaustive search. They can then be converted into alternate index types that have faster approximate search as described [here](https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-dense-vector-index).

## Creating dense indexes
We create one index for each of the 46 parts of the corpus separately at first. Each of these processes requires around 40-45 GB of RAM. The script `run_encoder_all.sh` starts these 46 processes - each using one GPU. The distribution of indexes across nodes is similar to the one for Lucene indexes. 0-11 is on 2-25, 12-23 on 2-27, 24-35 on 2-31, and 36-46 on 2-29. Since each of these nodes has 4 GPUs, a maximum of 4 processes run at once. Others remain queued. The script takes these arguments:
1. `-e`: the encoder name. Use the huggingface path or a local path to be used by the `transformers` library. 
1. `-p`: the type of pooling (`mean` or `cls` - using `mean` right now)
1. `-l`: the sequence length
1. `-o`: the directory in which to write the index directories (use an SSD path in the worker node)
1. `-s`: an indentifier that you can assign to an index (eg. "contriever512"). It is used to create the index name

Example usage:

```
bash run_encoder_all.sh -e facebook/contriever-msmarco -p mean -l 512 -o /ssd/hmehrotr -s contriever512
```

This will trigger jobs to create indexes. The name of an individual index - say for partition 03 - will be `CW22_ind_en00_03_contriever512_mean` and will be stored in the directory `/ssd/hmehrotr` on the respective nodes. Similarly, there will be indexes for the 46 partitions.

After doing this, indexes have to be merged to create the final 4 partitions. To do this, run `bash run_merge_indexes.sh`. This will create index `/ssd/hmehrotr/CW22_ind_en00_00_11_contrmarco_mean` on 2-25 and similarly for the other 3 nodes.

## Run search on dense indexes
The script `run_search.sh` contains commands to trigger 4 jobs - one to search on each of the index partitions. It will create 4 output files in the TREC format. Run `bash run_search.sh` to run batched search. The batch size can be edited inside the script using the `--batch-size` parameter in the commands. Currently, the commands use the set of 2110 evaluation queries in this repository. The `--hits` parameter controls the number of search results to return.

There is also a Python script `search_with_logging.py` to run search one query at a time on an index partition. We used it to get statistics on the latency of search using the 2110 evaluation queries. An example to run it on the 12-23 index partition is:

```
sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=80G --time=0 --nodelist=boston-2-27 \
  --wrap="python search_with_logging.py   --index /ssd/hmehrotr/CW22_ind_en00_12_23_contrmarco_mean   --encoder-class contriever --encoder facebook/contriever-msmarco   --topics /bos/tmp2/hmehrotr/query_3k.tsv --hits 1000   --device cpu   --output /bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_single_query_1k_metrics.tsv "
```

We found that the metrics for 1000 and 4000 hits were very similar. Example output is:

```
50th percentile query run time = 6.206277012825012 seconds
75th percentile query run time = 6.297831773757935 seconds
90th percentile query run time = 6.3599666357040405 seconds
95th percentile query run time = 6.436535763740539 seconds
99th percentile query run time = 6.475239789485931 seconds

50th percentile query search time = 6.203025937080383 seconds
75th percentile query search time = 6.29453980922699 seconds
90th percentile query search time = 6.356656241416931 seconds
95th percentile query search time = 6.433258473873138 seconds
99th percentile query search time = 6.4719913077354425 seconds
```

'query search time' is just the time taken by the Faiss search call while 'query run time' includes the time taken to write the result to the output file.

## Dense search service
You can run a Flask service with a search endpoint to perform search on a dense index. Navigate to the `dense` directory of the repository and run the following command:

```
sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=80G --time=0 --nodelist=boston-2-25 \
  --wrap="python dense_service.py   --index /ssd/hmehrotr/CW22_ind_en00_00_11_contrmarco_mean   --encoder-class contriever --encoder facebook/contriever-msmarco  --device cpu "
```

The above command would run the `dense_service.py` script which has the code for the search API on the note containing the 00-11 index partition. Similarly, 3 other search services can be started.

Now, from the first few lines of the slurm log file of the job running the service, one can check the IP and port of the service. If the IP is `10.1.1.28` and port `7001` then the API URL for the query "Who can introduce new bills" would be:

`http://10.1.1.28:7001/search?query=Who+can+introduce+new+bills&hits=1000`

Here the argument `hits` specifies the number of results to obtain from vector search (default is 1000). To test this API from command line from the boston cluster, run:

`curl --request GET "http://10.1.1.28:7001/search?query=Who+can+introduce+new+bills&hits=1000"`

The output is a JSON with a single key `results`. The value is an array of `result` objects in decreasing order of scores. Each `result` object has the following:

1. `docid`: String document id
1. `score`: Float vector similarity score

So, for 2 results, the output JSON would look like:

```
{
  "results": [
    {
      "docid": "ABCD1234",
      "score": 0.56
    },
    {
      "docid": "EFGH1234",
      "score": 0.40
    }
  ]
}
```



