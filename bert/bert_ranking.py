from flask import Flask, request
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
import datetime
from urllib.parse import urlparse
import gc
import aiohttp
import asyncio
import time
import argparse
from tqdm import tqdm

def dict_cuda(dictionary):
    tensor_keys = []
    for k, v in dictionary.items():
        if isinstance(v, torch.Tensor):
            tensor_keys.append(k)
    for k in tensor_keys:
        dictionary[k] = dictionary.pop(k).cuda()

base_urls = ["http://10.1.1.28:7001/search?", "http://10.1.1.29:7001/search?", "http://10.1.1.17:7001/search?", "http://10.1.1.30:7001/search?"]

async def get_results(session, url):
    async with session.get(url) as resp:
        results = await resp.json()
        return results['results']

async def main(q, hits):

    all_results = []

    async with aiohttp.ClientSession() as session:

        tasks = []
        for base_url in base_urls:
            url = base_url + "query=" + "+".join(q.split()) + "&hits=" + str(hits)
            tasks.append(asyncio.ensure_future(get_results(session, url)))

        shard_results = await asyncio.gather(*tasks)
        for shard_result in shard_results:
            all_results.extend(shard_result)

        return all_results

SCORE_IDX = 0
LENGTH_LIMIT = 512

parser = argparse.ArgumentParser()
parser.add_argument("--model-location", type=str, required=True)
parser.add_argument("--out-path", type=str, required=True)
parser.add_argument("--fs-hits", type=int, required=True)
parser.add_argument('--url', action='store_true')
parser.set_defaults(url=False)
args = parser.parse_args()

modelLocation = args.model_location
tokenizerLocation = modelLocation if "checkpoint" not in modelLocation else modelLocation[:modelLocation.index("checkpoint")-1]

config = AutoConfig.from_pretrained(modelLocation, num_labels=1)
model = AutoModelForSequenceClassification.from_pretrained(modelLocation, config=config)
tokenizer = AutoTokenizer.from_pretrained(tokenizerLocation, use_fast=True)
SEP = tokenizer.sep_token

model.cuda()
model.half()
model.eval()
torch.cuda.empty_cache()

qf = open("../query_3k.tsv","r")
resstr = ""
qn = 0



with open(args.out_path, "w") as fo:
    for line in tqdm(qf):
        qn+=1

        l = line.strip()
        qid = l.split('\t',1)[0]
        query = l.split('\t',1)[1]

        # start = time.time()

        hits= args.fs_hits

        query = query.replace("\n", "")

        all_results = asyncio.run(main(query, hits))

        if (len(all_results)!=4*hits):
            print ("Result count mismatch", query) 
        
        all_results = sorted(all_results, key=lambda r : r['score'], reverse=True)[:hits]

        # bm25_time = time.time()

        # print ("Using url: "+str(args.url))
        # print ("Till bm25 time: " + str(bm25_time-start))
    
        #BERT reranking
        pairs, docBatch, bert_results = [], [], []
        count = 1
        for doc in all_results:

            fulltext = doc["body"]
            fulltext = fulltext[:4000]
            url = doc["url"]
            doc["title"] = fulltext.split("\n")[0]
            if args.url:
                pairs.append((query, url + SEP + doc["title"] + SEP + fulltext))
            else:
                pairs.append((query, doc["title"] + SEP + fulltext))
            docBatch.append(doc)

            if count % 256 == 0 or count == len(all_results):
                torchStart = datetime.datetime.now()
                encoded_batch = tokenizer.batch_encode_plus(
                    pairs,
                    truncation=True,
                    truncation_strategy='only_second',
                    padding='max_length',
                    max_length=LENGTH_LIMIT,
                    return_tensors='pt'
                )
                dict_cuda(encoded_batch)

                # print(encoded_batch['input_ids'][:2])

                with torch.no_grad():
                    output = model(**encoded_batch)
                    scores = output[0].cpu().numpy()[:, SCORE_IDX].tolist()
                for doc_dict, score in zip(docBatch, scores):
                    #app.logger.info(score)
                    doc_dict['bert_score'] = score
                    #app.logger.info(doc_dict['score'])
                    bert_results.append((doc_dict, score))

                del pairs
                del docBatch
                del encoded_batch
                del scores
                gc.collect()
                pairs, docBatch = [], []
                torch.cuda.empty_cache()
            count += 1
            

        bertResults = sorted(bert_results, key=lambda x: (x[1],x[0]['score']), reverse=True)

        bert_time = time.time()


        for i, bert_result_tup in enumerate(bertResults):
            bert_result = bert_result_tup[0]
            resstr = str(qid) + " Q0 " + str(bert_result["docid"]) + " " + str(i+1) + " " + str(bert_result["bert_score"]) + " Bert\n"
            fo.write(resstr)
        
        write_time = time.time()
        # print (query)
        # print ("Till bm25 time: " + str(bm25_time-start))
        # print ("Till bert time: " + str(bert_time-start))
        # print ("Till write time: " + str(write_time-start))
        # print (resstr[-150:])
        # print ()

qf.close()