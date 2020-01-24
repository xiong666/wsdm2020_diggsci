#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
import os
import re
from tqdm import tqdm
import pickle
from utils import pre_process, digest, get_n_max
from gensim.summarization import bm25
import multiprocessing as mlp
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
import argparse

parser = argparse.ArgumentParser(description='Get recall using bm25 algorithm')
parser.add_argument('-n', '--number', default=200, type=int,
                    help='numbers to recall')
parser.add_argument('-data_dir', type=str,
                    help='data directory stores .csv files')
args = parser.parse_args()

n = args.n
data_path = args.data_dir

weights = (2, 2, 1)
n_cpu = mlp.cpu_count() - 1


logging.info('get_bm25_result...')

train_path = os.path.join(data_path, 'train_release.csv')
valid_path = os.path.join(data_path, 'validation.csv')
test_path = os.path.join(data_path, 'test.csv')
cand_path = os.path.join(data_path, 'candidate_paper_for_wsdm2020.csv')

train_data = pd.read_csv(train_path)
valid_data = pd.read_csv(valid_path)
test_data = pd.read_csv(test_path)
cand_data = pd.read_csv(cand_path)

cand_data['abstract'] = cand_data['abstract'].fillna('no_content')
cand_data['title'] = cand_data['title'].fillna('no_content')
cand_data['keywords'] = cand_data['keywords'].fillna('no_content')

cand_data = cand_data.dropna(subset=['paper_id'])
print('cand_data.shape', cand_data.shape)
cand_data = cand_data.reset_index(drop=True)


cand_data['keywords_title_abstract'] = cand_data[['title', 'abstract', 'keywords']].apply(
    lambda x: (x['keywords'] + ' ') * weights[0] + (x['title'] + ' ') * weights[1] + x['abstract'] *
              weights[2], axis=1)

raw_corpus = np.concatenate((train_data['description_text'].values,
                             test_data['description_text'].values,
                             cand_data['keywords_title_abstract'].values))

if not os.path.exists('corpus.pkl'):
    corpus = []
    pool = mlp.Pool(processes=n_cpu)
    for y in tqdm(pool.imap(pre_process, raw_corpus), desc='tokenzing', total=len(raw_corpus)):
        corpus.append(y)
    with open('corpus.pkl', 'wb') as fw:
        pickle.dump(corpus, fw)
else:
    with open('corpus.pkl', 'rb') as fr:
        corpus = pickle.load(fr)

afore_len = len(train_data['description_text'].values) + len(test_data['description_text'].values)
cand_corpus = corpus[afore_len:]

del raw_corpus

# bm25 model
bm25Model = bm25.BM25(cand_corpus)


def get_scores(query):

    score = bm25Model.get_scores(query)
    return get_n_max(score, n)


if not os.path.exists('test_bm25_{}.csv'.format(n)):

    query_list_test = test_data['description_text'].apply(
        lambda x: str(x).replace('\xa0', ' ')).apply(digest).apply(
        lambda x: pre_process(re.sub(r'[\[|,]+\*\*\#\#\*\*[\]|,]+', '', str(x)))).values

    # predict for test
    scores_idx_test = np.zeros((len(query_list_test), n))

    pool = mlp.Pool(processes=n_cpu)
    for i, y in tqdm(enumerate(pool.imap(get_scores, query_list_test)),
                     desc='getting scores', total=len(query_list_test)):
        scores_idx_test[i] = y

    test_paper_predict = pd.DataFrame(
        cand_data['paper_id'].iloc[scores_idx_test.reshape(-1, ).astype('int')].values.reshape(-1, n))

    test_bm25 = pd.merge(test_data[['description_id']],
                            test_paper_predict, left_index=True, right_index=True)

    test_bm25.to_csv('test_bm25_{}.csv'.format(n), index=False)


if not os.path.exists('train_bm25_{}.csv'.format(n)):

    query_list_train = train_data['description_text'].apply(
        lambda x: str(x).replace('\xa0', ' ')).apply(digest).apply(
        lambda x: pre_process(re.sub(r'[\[|,]+\*\*\#\#\*\*[\]|,]+', '', str(x)))).values

    # predict for train
    scores_idx_train = np.zeros((len(query_list_train), n))

    pool = mlp.Pool(processes=n_cpu)
    for i, y in tqdm(enumerate(pool.imap(get_scores, query_list_train)),
                     desc='getting scores', total=len(query_list_train)):
        scores_idx_train[i] = y
    train_paper_predict = pd.DataFrame(
        cand_data['paper_id'].iloc[scores_idx_train.reshape(-1, ).astype('int')].values.reshape(-1, n))

    train_bm25 = pd.merge(train_data[['description_id']],
                            train_paper_predict, left_index=True, right_index=True)

    train_bm25.to_csv('train_bm25_{}.csv'.format(n), index=False)
