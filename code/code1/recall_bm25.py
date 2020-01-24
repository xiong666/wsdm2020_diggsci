import re
from gensim.summarization import bm25
from multiprocessing import Process,cpu_count,Manager,Pool
from sklearn.externals import joblib
from gensim import corpora,similarities,models
import pandas as pd
from tqdm import tqdm_notebook
from util import *
from  tqdm import  tqdm
tqdm.pandas()
import  numpy as np
import pickle
import  os
import warnings
import time
import  h5py
warnings.filterwarnings('ignore')


model_path='./'

dictionary = corpora.Dictionary.load('{}train_dictionary2.dict'.format(model_path))
index = similarities.SparseMatrixSimilarity.load('{}train_index2.index'.format(model_path))
item_id_list = joblib.load('{}paper_id2.pkl'.format(model_path))

with open('{}train_content2.pkl'.format(model_path),'rb') as fr:
    corpus = pickle.load(fr)

####bm5模型
print('get bm5')
bm25Model = bm25.BM25(corpus)
average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())


print('模型加载完成')


def get_recall_number(val, n):
    docs=val['description_text'].values
    ids=val['description_id'].values
    submit = np.zeros((len(docs), 2 * n+1)).astype(np.str)
    count = len(docs)
    bar = tqdm(range(count))
    for i in bar:
        doc = docs[i]
        id = ids[i]
        # 计算得分
        scores = np.array(bm25Model.get_scores(doc, average_idf))
        related_doc_indices = scores.argsort()[:-n-1:-1]
        col=[id]+[item_id_list[index] for index in related_doc_indices] + [scores[index] for index in related_doc_indices]
        #保存得分序列
        submit[i]=col
    return submit



'多进程处理'
def pool_extract(data, f ,n,chunk_size, worker=5):
    cpu_worker = os.cpu_count()
    print('cpu 核心有：{}'.format(cpu_worker))
    if worker == -1 or worker > cpu_worker:
        worker = cpu_worker
    print('使用cpu:{}'.format(worker))
    t1 = time.time()
    len_data = len(data)
    start = 0
    end = 0
    p = Pool(worker)
    res = []  # 保存的每个进程的返回值
    while end < len_data:
        end = start + chunk_size
        if end > len_data:
            end = len_data
        rslt = p.apply_async(f, (data[start:end],n))
        start = end
        res.append(rslt)
    p.close()
    p.join()
    t2 = time.time()
    print((t2 - t1)/60)
    results = np.concatenate([i.get() for i in res], axis=0)
    return results





##############################测试集
if __name__=='__main__':
    valid = pd.read_csv('./data/stage2_test_pre.csv')
	
    valid = valid[valid['key_text_pre'].notnull()].reset_index(drop=True)
	
    #获取关键句
    valid['key_text_pre'] = valid['key_text_pre'].apply(lambda x: x.split(' '))
    valid['key_text_pre_len'] = valid['key_text_pre'].apply(lambda x: len(x))
    valid.loc[valid['key_text_pre_len'] < 7, 'key_text_pre'] = valid.loc[valid['key_text_pre_len'] < 7][
        'description_text'].apply(
        lambda x: pre_process(re.sub(r'[\[|,]+\*\*\#\#\*\*[\]|,]+','',x))).values

    #描述id
    ids = list(valid['description_id'].values)
    #描述内容
    docs = list(valid['key_text_pre'].values)
    print(valid.shape)

    valid=pd.DataFrame({'description_id':ids,'description_text':docs})

    #多进程召回
    submit=pool_extract(valid, get_recall_number, 200, 3000, worker=4)
    df = pd.DataFrame(submit)
    df.to_csv('{}stage2_test_pairs_200number2_with_scores.csv'.format(model_path),header=None,index=False)


    
    #############train
    train = pd.read_csv('./data/train_pre.csv')
    # 获取关键句
    train['key_text_pre'] = train['key_text_pre'].apply(lambda x: x.split(' ') if str(x)!='nan' else 'none')
    train['key_text_pre_len'] = train['key_text_pre'].apply(lambda x: len(x))
    train.loc[train['key_text_pre_len'] < 5, 'key_text_pre'] = train.loc[train['key_text_pre_len'] < 5][
        'description_text'].apply(
        lambda x: pre_process(re.sub(r'[\[|,]+\*\*\#\#\*\*[\]|,]+', '', x))).values

    # 描述id
    ids = list(train['description_id'].values)
    # 描述内容
    docs = list(train['key_text_pre'].values)


    train = pd.DataFrame({'description_id': ids, 'description_text': docs})

    #召回
    submit = pool_extract(train, get_recall_number, 100, 5000, worker=4)
    df = pd.DataFrame(submit)
    df.to_csv('{}train_pairs_100number2_scores.csv'.format(model_path), header=None, index=False)
