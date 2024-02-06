from transformers import AutoTokenizer
import json
import os
from collections import defaultdict
import datasets
from tqdm import tqdm
from argparse import ArgumentParser
import csv

parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
# parser.add_argument('--rank_file', required=True)
parser.add_argument('--truncate', type=int, default=512)
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--end', type=int, required=True)
parser.add_argument('--partition', type=int, required=True)

# parser.add_argument('--sample_from_top', type=int, required=True)
# parser.add_argument('--n_sample', type=int, default=100)
# parser.add_argument('--random', action='store_true')
# parser.add_argument('--json_dir', required=True)

# parser.add_argument('--qrel', required=True)
# parser.add_argument('--query_collection', required=True)
# parser.add_argument('--doc_collection', required=True)
args = parser.parse_args()


root = "/bos/tmp3/cx/AnchorDR_data_release/"
qrel_path = root + "qrels.train.tsv"
corpus_path = root + "corpus.tsv"
negs_path = root + "train.BM25.negatives.tsv"
queries_path = root + "queries.train.tsv"

def read_qrel():
    qrel = {}
    with open(qrel_path, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, _, docid, rel] in tsvreader:
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]
    return qrel

def read_negs():
    negs = {}
    with open(negs_path, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, docids] in tsvreader:
            if topicid in negs:
                negs[topicid].extend(docids.split(','))
            else:
                negs[topicid] = docids.split(',')
    return negs

def get_doc_map():
    doc_map = {}
    with open(corpus_path, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        idx = 0
        for [did, _, _] in tsvreader:
            doc_map[str(did)] = idx
            idx+=1
    return doc_map

def get_qry_map():
    qry_map={}
    with open(queries_path, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        idx = 0
        for [qid, _] in tsvreader:
            qry_map[str(qid)] = idx
            idx+=1
    return qry_map

qrel = read_qrel()
negs = read_negs()

no_judge = set(qrel.keys()).difference(set(negs.keys()))

print(f'{len(no_judge)} queries have no negatives and skipped', flush=True)

columns = ['did', 'title', 'body']

collection = datasets.load_dataset(
    'csv',
    data_files=corpus_path,
    column_names=['did', 'title', 'body'],
    delimiter='\t',
    ignore_verifications=True,
)['train']

qry_collection = datasets.load_dataset(
    'csv',
    data_files=queries_path,
    column_names=['qid', 'qry'],
    delimiter='\t',
    ignore_verifications=True,
)['train']

doc_map = get_doc_map() #{str(x['did']): idx for idx, x in enumerate(collection)}
qry_map = get_qry_map() # {str(x['qid']): idx for idx, x in enumerate(qry_collection)}

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

#out_file = '/bos/tmp2/hmehrotr/bert/train_data.json'
out_file = '/bos/tmp2/hmehrotr/bert/train_data/part_'+str(args.partition)+'.json'

queries = list(negs.keys())

i=0

with open(out_file, 'w') as f:
    for qid in tqdm(queries[args.start-1:args.end]):

        # i+=1

        # if i<args.start or i>args.end:
        #     continue

        neg_encoded = []

        for neg in negs[qid]:
            idx = doc_map[neg]
            item = collection[idx]
            did, title, body = (item[k] for k in columns)
            title, body = map(lambda v: v if v else '', [title, body])
            encoded_neg = tokenizer.encode(
                title + tokenizer.sep_token + body,
                add_special_tokens=False,
                max_length=args.truncate,
                truncation=True
            )
            neg_encoded.append({
                'passage': encoded_neg,
                'pid': neg,
            })

        pos_encoded = []

        for pos in qrel[qid]:
            idx = doc_map[pos]
            item = collection[idx]
            did, title, body = (item[k] for k in columns)
            title, body = map(lambda v: v if v else '', [title, body])
            encoded_pos = tokenizer.encode(
                title + tokenizer.sep_token + body,
                add_special_tokens=False,
                max_length=args.truncate,
                truncation=True
            )
            pos_encoded.append({
                'passage': encoded_pos,
                'pid': pos,
            })

        q_idx = qry_map[qid]
        query_dict = {
            'qid': qid,
            'query': tokenizer.encode(
                qry_collection[q_idx]['qry'],
                add_special_tokens=False,
                max_length=args.truncate,
                truncation=True),
        }
        item_set = {
            'qry': query_dict,
            'pos': pos_encoded,
            'neg': neg_encoded,
        }

        f.write(json.dumps(item_set) + '\n')