#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from gensim.utils import tokenize as gen_tokenize
from nltk import word_tokenize, pos_tag
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import unicodedata

tags = {'.', '*', '!', ',', '(', ')', '[', ']', '{', '}', '?', '#', '$', '@', '**', '"', ':', "''", '``', '\u200b',
        ';', '%', '^', '...', '&', ' ', 'figure', 'figure1', 'figure2', 'figure3', 'figure4', 'figure5',
        'no_content'}
stop_words = set(stopwords.words('english')) | tags


def tokenize(text, tokenizer='nltk'):
    text = re.sub(r'\s+', ' ', str(text))
    if tokenizer.lower() == 'nltk':
        token_words = word_tokenize(text)  # 输入的是列表
    elif tokenizer.lower() == 'gensim':
        token_words = list(gen_tokenize(text))
    elif tokenizer.lower() == 'all':
        token_words = word_tokenize(text)

        # # bigram
        # text = re.sub(r'\s+', ' ', str(text))
        # bigram = [a + ' ' + b for a, b in zip(token_words[:-1], token_words[1:])]

        token_words.extend(list(gen_tokenize(text)))

        # token_words.extend(bigram)
        # token_words.extend(text.split(' '))
    else:
        raise Exception('Tokenizer must be nltk or gensim or all')
    token_words = pos_tag(token_words)
    return token_words


def stem(token_words, stemmer=None):
    '''
        词形归一化
    '''
    if not stemmer:
        words_lematizer = [word for word, tag in token_words]
    elif stemmer.lower() == 'wordnet':
        wordnet_lematizer = WordNetLemmatizer()  # 单词转换原型
        words_lematizer = []
        for word, tag in token_words:
            if tag.startswith('NN'):
                word_lematizer = wordnet_lematizer.lemmatize(word, pos='n')  # n代表名词
            elif tag.startswith('VB'):
                word_lematizer = wordnet_lematizer.lemmatize(word, pos='v')  # v代表动词
            elif tag.startswith('JJ'):
                word_lematizer = wordnet_lematizer.lemmatize(word, pos='a')  # a代表形容词
            elif tag.startswith('R'):
                word_lematizer = wordnet_lematizer.lemmatize(word, pos='r')  # r代表代词
            else:
                word_lematizer = wordnet_lematizer.lemmatize(word)
            words_lematizer.append(word_lematizer)
    elif stemmer.lower() == 'spacy':
        nlp = spacy.load('en_core_web_sm')
        token_words = [word for word, tag in token_words]
        words_lematizer = [token.lemma_ for token in nlp(' '.join(token_words))]
    elif stemmer.lower() == 'all':
        wordnet_lematizer = WordNetLemmatizer()  # 单词转换原型
        words_lematizer = []

        # nltk
        for word, tag in token_words:
            if tag.startswith('NN'):
                word_lematizer = wordnet_lematizer.lemmatize(word, pos='n')  # n代表名词
            elif tag.startswith('VB'):
                word_lematizer = wordnet_lematizer.lemmatize(word, pos='v')  # v代表动词
            elif tag.startswith('JJ'):
                word_lematizer = wordnet_lematizer.lemmatize(word, pos='a')  # a代表形容词
            elif tag.startswith('R'):
                word_lematizer = wordnet_lematizer.lemmatize(word, pos='r')  # r代表代词
            else:
                word_lematizer = wordnet_lematizer.lemmatize(word)
            words_lematizer.append(word_lematizer)


        token_words = [word for word, tag in token_words]
        # # spacy
        # nlp = spacy.load('en_core_web_sm')
        # words_lematizer.extend([token.lemma_ for token in nlp(' '.join(token_words))])
        words_lematizer.extend(token_words)
    else:
        raise Exception('stemmer must be wordnet or spacy or None')
    return words_lematizer

def is_number(s):
    '''
        判断字符串是否为数字
    '''
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def pre_process(text, tokenizer='all', stemmer='wordnet'):
    '''
        文本预处理
    '''
    token_words = tokenize(text, tokenizer)
    token_words = stem(token_words, stemmer)
    token_words = [word.lower() for word in token_words if word.lower() not in stop_words and not is_number(word)]

    return token_words


def get_n_max(score, n):
    '''
    get 1st to nth largest numbers' indexes
    :param score: shape: (x,) the score array
    :param n: nth largest number index to get
    :return: 1st to nth largest numbers' indexes
    '''
    score = np.array(score).reshape(-1, )
    nidx = np.argpartition(score, -n, axis=-1)[-n:]
    return nidx[np.argsort(score[nidx])][::-1]


def digest(text):
    text = str(text)
    text = text.replace('al.', '').split('. ')
    t=''
    pre_text=[]
    len_text=len(text)-1
    add=True
    pre=''
    while len_text>=0:
        index=text[len_text]
        index+=pre
        if len(index.split(' '))<=3 :
            add=False
            pre=index+pre
        else:
            add=True
            pre=''
        if add:
            pre_text.append(index)
        len_text-=1
    if len(pre_text)==0:
        pre_text=text
    pre_text.reverse()
    for index in pre_text:
        if index.find('[**##**]') != -1:
            index = re.sub(r'[\[|,]+\*\*\#\#\*\*[\]|,]+','',index)
            index+='. '
            t+=index
    return t


def get_paper_sentence(text):
    # get the referenced sentence from the paragraph
    sents = sent_tokenize(text)
    for idx, sent in enumerate(sents):
        if '[**##**]' in sent:
            # if idx >= 1:
            #     return '.'.join(sents[idx - 1: idx + 1]).lower().strip().replace('[**##**]', ' ')
            if (len(sent.strip()) <= 40) and (idx >= 1):
                return '.'.join(sents[idx - 1: idx + 1]).lower().strip().replace('[**##**]', ' ')
            else:
                return sent.lower().strip().replace('[**##**]', ' ')


def map_score(paper_predict, n, train_data):
    map_score = 0
    total_score = 0
    indices = {}
    n = min(n, paper_predict.shape[1])
    for idx, paper in enumerate(train_data['paper_id'].values):
        for i, pred in enumerate(paper_predict[idx][:n]):
            if paper == pred:
                map_score += 1 / (i + 1)
                total_score += 1
                indices.setdefault(i+1, []).append(idx)
    map_score = map_score / len(train_data)
    recall_rate = total_score / len(train_data)
    print(map_score, recall_rate)
    return map_score, recall_rate



