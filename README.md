# cw22b_en_search

This repository contains the code and documentation for an independent study project done under the supervision of Prof. Jamie Callan at Carnegie Mellon University in Fall 2023. The ClueWeb22 dataset containing nearly 10 billion documents was released in 2022 to support academic and industry research. The goal of this project was to build retrieval baselines for the English section of the "super head" part (category B) of this dataset. These baselines can then be used by the research community to compare their systems and also to generate data to train/evaluate new retrieval and ranking algorithms. The repository covers sparse and dense first stage retrievals as well as neural rerankers that were implemented for this dataset.

The repository is organized as follows:

1. `lucene`: Directory with code and documentation of our BM25 implementation and service. Has its own README for documentation.
2. `dense`: Directory with code and documentation for indexing and searching dense Faiss indexes and hosting a search service. Has its own README for documentation.
3. `bert`: Directory with code and documentation to train rerankers. Has its own README for documentation.
4. `query_3k.tsv`: A set of 2110 queries we used for evaluation.
5. `qrel_3k.tsv`: Relevance judgements on the 2110 queries.
6. `trec_eval-9.0.7`: A copy of the trec_eval software available [here](https://trec.nist.gov/trec_eval/). Instructions to use it are present [here](http://www.rafaelglater.com/en/post/learn-how-to-use-trec_eval-to-evaluate-your-information-retrieval-system).

Project report - https://arxiv.org/pdf/2402.04357.pdf
