#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import os
from typing import OrderedDict

from tqdm import tqdm

from pyserini.search import FaissSearcher, BinaryDenseSearcher, TctColBertQueryEncoder, QueryEncoder, \
    DprQueryEncoder, BprQueryEncoder, DkrrDprQueryEncoder, AnceQueryEncoder, AggretrieverQueryEncoder, AutoQueryEncoder, DenseVectorAveragePrf, \
    DenseVectorRocchioPrf, DenseVectorAncePrf#, OpenAIQueryEncoder

from pyserini.encode import PcaEncoder
from pyserini.search.lucene import LuceneSearcher

import time

from flask import Flask, request
from flask import jsonify


# from ._prf import DenseVectorAveragePrf, DenseVectorRocchioPrf

# Fixes this error: "OMP: Error #15: Initializing libomp.a, but found libomp.dylib already initialized."
# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = Flask(__name__)

searcher = None


def define_dsearch_args(parser):
    parser.add_argument('--index', type=str, metavar='path to index or index name', required=True,
                        help="Path to Faiss index or name of prebuilt index.")
    parser.add_argument('--encoder-class', type=str, metavar='which query encoder class to use. `default` would infer from the args.encoder',
                        required=False,
                        choices=["dkrr", "dpr", "bpr", "tct_colbert", "ance", "sentence", "contriever", "auto", "aggretriever", "openai-api"],
                        default=None,
                        help='which query encoder class to use. `default` would infer from the args.encoder')
    parser.add_argument('--encoder', type=str, metavar='path to query encoder checkpoint or encoder name',
                        required=False,
                        help="Path to query encoder pytorch checkpoint or hgf encoder model name")
    parser.add_argument('--tokenizer', type=str, metavar='name or path',
                        required=False,
                        help="Path to a hgf tokenizer name or path")
    parser.add_argument('--device', type=str, metavar='device to run query encoder', required=False, default='cpu',
                        help="Device to run query encoder, cpu or [cuda:0, cuda:1, ...]")
    parser.add_argument('--query-prefix', type=str, metavar='str', required=False, default=None,
                        help="Query prefix if exists.")
    parser.add_argument('--searcher', type=str, metavar='str', required=False, default='simple',
                        help="dense searcher type")
    parser.add_argument('--max-length', type=int, help='max length', default=256, required=False)
    parser.add_argument('--ef-search', type=int, metavar='efSearch for HNSW index', required=False, default=None,
                        help="Set efSearch for HNSW index")


def init_query_encoder(encoder, encoder_class, tokenizer_name, device, prefix, max_length):
    encoded_queries_map = {
        'msmarco-passage-dev-subset': 'tct_colbert-msmarco-passage-dev-subset',
        'dpr-nq-dev': 'dpr_multi-nq-dev',
        'dpr-nq-test': 'dpr_multi-nq-test',
        'dpr-trivia-dev': 'dpr_multi-trivia-dev',
        'dpr-trivia-test': 'dpr_multi-trivia-test',
        'dpr-wq-test': 'dpr_multi-wq-test',
        'dpr-squad-test': 'dpr_multi-squad-test',
        'dpr-curated-test': 'dpr_multi-curated-test'
    }
    encoder_class_map = {
        "dkrr": DkrrDprQueryEncoder,
        "dpr": DprQueryEncoder,
        "bpr": BprQueryEncoder,
        "tct_colbert": TctColBertQueryEncoder,
        "ance": AnceQueryEncoder,
        "sentence": AutoQueryEncoder,
        "contriever": AutoQueryEncoder,
        "aggretriever": AggretrieverQueryEncoder,
        # "openai-api": OpenAIQueryEncoder,
        "auto": AutoQueryEncoder,
    }

    if encoder:
        _encoder_class = encoder_class

        # determine encoder_class
        if encoder_class is not None:
            encoder_class = encoder_class_map[encoder_class]
        else:
            # if any class keyword was matched in the given encoder name,
            # use that encoder class
            for class_keyword in encoder_class_map:
                if class_keyword in encoder.lower():
                    encoder_class = encoder_class_map[class_keyword]
                    break

            # if none of the class keyword was matched,
            # use the AutoQueryEncoder
            if encoder_class is None:
                encoder_class = AutoQueryEncoder

        # prepare arguments to encoder class
        kwargs = dict(encoder_dir=encoder, tokenizer_name=tokenizer_name, device=device, prefix=prefix)
        if (_encoder_class == "sentence") or ("sentence" in encoder):
            kwargs.update(dict(pooling='mean', l2_norm=True))
        if (_encoder_class == "contriever") or ("contriever" in encoder):
            kwargs.update(dict(pooling='mean', l2_norm=False))
        if (_encoder_class == "openai-api") or ("openai" in encoder):
            kwargs.update(dict(max_length=max_length))
        return encoder_class(**kwargs)

    raise ValueError(f'No encoder found')

@app.route('/search', methods=['GET'])
def search():

    args = request.args
    query = args.get("query")

    if query is None:
        return "Missing query", 400

    hits = args.get("hits")

    if hits:
        try:
            hits = int(hits)
        except:
            hits=1000
    else:
        hits=1000

    single_query_start = time.time()

    if searcher is None:
        return "Searcher not initialized", 503

    results = searcher.search(query, hits)
    single_query_search_end = time.time()

    print ("Done in "+str(single_query_search_end-single_query_start)+" seconds")

    response = {"results": [{"docid": str(result.docid), "score": float(result.score)} for result in results]}

    return jsonify(response)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search a Faiss index.')
    # For some test collections, a query is doc from the corpus (e.g., arguana in BEIR).
    # We want to remove the query from the results. This is equivalent to -removeQuery in Java.
    # parser.add_argument('--remove-query', action='store_true', default=False, help="Remove query from results list.")

    define_dsearch_args(parser)
    args = parser.parse_args()

    query_encoder = init_query_encoder(
        args.encoder, args.encoder_class, args.tokenizer, args.device, args.query_prefix, args.max_length)

    # kwargs = {}

    searcher_init_start = time.time()

    if os.path.exists(args.index):
        searcher = FaissSearcher(args.index, query_encoder)
    else:
        searcher = FaissSearcher.from_prebuilt_index(args.index, query_encoder)

    if args.ef_search:
        searcher.set_hnsw_ef_search(args.ef_search)

    searcher_init_end = time.time()

    print ("Searcher initialized in "+str(searcher_init_end-searcher_init_start)+" seconds")

    if not searcher:
        exit()

    app.run(host='0.0.0.0', port=7001)
     