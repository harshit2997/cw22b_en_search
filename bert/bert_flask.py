from flask import Flask, request
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
import simplejson
import urllib.request
import datetime
from urllib.parse import urlparse
import random
import gc
import aiohttp
import asyncio
import time

app = Flask(__name__, static_url_path='')

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

#modelLocation = '/bos/tmp2/cmw2/BERT/fold1'
modelLocation = '/bos/usr0/mturk/BERT_service/models/bert_model_marco_cw09_mturk'

config = AutoConfig.from_pretrained(modelLocation, num_labels=1)
model = AutoModelForSequenceClassification.from_pretrained(modelLocation, config=config)
tokenizer = AutoTokenizer.from_pretrained(modelLocation, do_lower_case=True)
SEP = tokenizer.sep_token

model.cuda()
model.half()
model.eval()
torch.cuda.empty_cache()

@app.route('/search/bert', methods=['GET'])
def search():

    start = time.time()

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

    query = query.replace("\n", "")

    all_results = asyncio.run(main(query, hits))

    if (len(all_results)!=4*hits):
        return "Result count mismatch", 503  
    
    all_results = sorted(all_results, key=lambda r : r['score'], reverse=True)[:hits]

    bm25_time = time.time()
  
    #BERT reranking
    pairs, docBatch, bert_results = [], [], []
    count = 1
    for doc in all_results:

        fulltext = doc["body"]
        fulltext = fulltext[:4000]
        url = doc["url"]
        doc["title"] = fulltext.split("\n")[0]
        pairs.append((query, SEP + url + SEP + doc["title"] + SEP + fulltext))
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

            with torch.no_grad():
                output = model(**encoded_batch)
                scores = output[0].cpu().numpy()[:, SCORE_IDX].tolist()
            #scores = torch.softmax(model(**encoded_batch)[0])[:, SCORE_IDX].cpu().tolist()
            #app.logger.info('Scores: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            #app.logger.info(scores)
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
        
    
    #app.logger.info('Doc Dict: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #for result in results:
    #    app.logger.info('Result scores: ' + str(result[1]) + ' - ' + str(result[0]['score']))
    bertResults = sorted(bert_results, key=lambda x: (x[1],x[0]['score']), reverse=True)
    numFilteredResults = len(all_results)
    numBertResults = len(bertResults)
    app.logger.info("# of original results: " + str(numFilteredResults))
    app.logger.info("# of BERT results: " + str(numBertResults))

    #Filter out urls that are the same netloc
    # top10 =[]
    # url_dict = {}
    # num_not_filtered = 0
    # for result_dict in bertResults:
    #     url = result_dict[0]["url"]
    #     bm25_score = result_dict[0]["score"]
    #     #result_dict[0]["score"] = result_dict[1]
    #     parsedUrl = urlparse(url)
    #     netloc = parsedUrl.netloc
    #     if url_dict.get(netloc, None) == None:
    #         url_dict[netloc] = []
    #     url_dict[netloc].append(result_dict[0]["id"])
    #     if len(url_dict[netloc]) <= 10:
    #         num_not_filtered+=1
    #     elif (bm25_score in x["score"] for x["score"] in top10):
    #         result_dict[0]['filtered']=True
    #         app.logger.info("Duplicate document: " + result_dict[0]['id'])
    #     else:
    #         result_dict[0]['filtered']=True
    #     top10.append(result_dict)
    #     if num_not_filtered == 10:
    #         break

    # #Add highlights to results
    # #highlight_dict = response["highlighting"]
    # #app.logger.info("Highlight dict len " + str(len(highlight_dict)))
    # resultDictList = []
    # for topResult in top10:
    #     #app.logger.info(topResult[0])
    #     currentId = topResult[0]['id']
    #     if currentId.startswith(prefix):
    #         currentId = currentId[len(prefix):]
    #     highlight = highlight_dict[currentId]["fulltext"][0]
    #     #highlight = topResult[0]['fulltext'][:2000]
    #     score_string = str(topResult[0]['bert_score']) + " - " + str(topResult[0]['score'])
    #     result_dict = {'docId': topResult[0]['id'], 'title': topResult[0]['title'], 'url': topResult[0]['url'], 'highlight': highlight, 'score': score_string, 'filtered': topResult[0]['filtered']}
    #     resultDictList.append((result_dict))

    # if not is_interleave:
    #     resultDictList = insertRandom(
    #         resultDictList=resultDictList,
    #         randomDocList=randomDocList,
    #         random_docs = random_docs,
    #     )
    #     random.shuffle(resultDictList)

    #app.logger.info(len(resultDictList))
    # jsonString = simplejson.dumps(resultDictList) + '\n'
    jsonString = simplejson.dumps(bertResults[:10])

    bert_time = time.time()
    app.logger.info("Till bm25 time: " + str(bm25_time-start))
    app.logger.info("Till bert time: " + str(bert_time-start))
    return jsonString
			
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7001)
