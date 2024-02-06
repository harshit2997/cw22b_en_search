import pandas as pd
import time

print ("Started")
ts = time.time()
qpath = "../../query_3k.tsv"
qrel_path = "../../qrel_3k.tsv"
res_paths = ["/bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_t512_d768_00_11.tsv", \
             "/bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_t512_d768_12_23.tsv", \
             "/bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_t512_d768_24_35.tsv", \
             "/bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_t512_d768_36_46.tsv",]
             #"/bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_t512_d768_0004.tsv", \
             #"/bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_t512_d768_0005.tsv", \
             #"/bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_t512_d768_0006.tsv", \
             #"/bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_t512_d768_0007.tsv", \
             #"/bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_t512_d768_0008.tsv", \
             #"/bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_t512_d768_0009.tsv", \
             #"/bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_t512_d768_0010.tsv", \
             #"/bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_t512_d768_0011.tsv"]

df_list = []
qids = []

with open(qpath, 'r') as f:
  for l in f:
    qids.append(l.strip().split('\t')[0])

for res_path in res_paths:
    df = pd.read_csv(res_path, sep=' ', names = ['qid', 'mock', 'docid', 'rank', 'score', 'runid'], dtype=str)
    df_list.append(df)

all_df = pd.concat(df_list)
all_df = all_df.sort_values(by=['qid', 'score', 'docid'], ascending=[True, False, True])
all_df['rank'] = all_df.groupby(['qid']).cumcount()+1
all_df = all_df.groupby('qid').head(4000).reset_index(drop=True)

print (len(all_df))

result_qids = all_df['qid'].unique()
print (set(qids).difference(set(result_qids)))
print (set(result_qids).difference(set(qids)))

for i in range(len(qids)):
   if not qids[i] == result_qids[i]:
      print (qids[i], result_qids[i])

all_df.to_csv("/bos/tmp2/hmehrotr/CW22_dense_res/test/contriever_marco_t512_d768_all.tsv", sep=' ', index=False)
b= time.time()
print (str(b-a))
