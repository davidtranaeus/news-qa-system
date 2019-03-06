# from corenlp import StanfordNLP
import json
from pprint import pprint
from textblob import TextBlob
from nltk import Tree
import spacy
from spacy.tokenizer import Tokenizer
import nltk.data
import re
from nltk import sent_tokenize
from data_processing import *
import torch
from InferSent.models import InferSent
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.util import bigrams

def init_infersent():
  V = 1
  # MODEL_PATH = 'InferSent/encoder/infersent%s.pkl' % V
  MODEL_PATH = 'InferSent/encoder/infersent1.pickle'
  params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                  'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
  infersent = InferSent(params_model)
  infersent.load_state_dict(torch.load(MODEL_PATH))

  W2V_PATH = 'InferSent/dataset/GloVe/glove.840B.300d.txt'
  infersent.set_w2v_path(W2V_PATH)
  return infersent

def pg_infersent_acc():
  dp = DataProcessor(data_set="squad")
  dp.load()

  infersent = init_infersent()

  sentences = []
  for i in dp.articles:
    for j in i.sentences:
      sentences.append(j.text)
    for j in i.questions:
      for k in j.a:
        sentences.append(k)
      for k in j.q:
        sentences.append(k)

  print("Building vocab")
  # infersent.build_vocab_k_words(K=100000)
  infersent.build_vocab(sentences, tokenize=True)
  print("Done")

  results = [0,0]
  
  # for i in dp.articles:
  for c,i in enumerate(dp.articles[:100]):
    print(c)

    s_embeddings = infersent.encode([s.text for s in i.sentences])

    for j in i.questions:
      q_embedding = infersent.encode([j.q])
      scores = cosine_similarity(s_embeddings, q_embedding)
      if np.argmax(scores) == j.correct_sentence:
      # if j.correct_sentence in scores.argsort()[-3:][::-1]:
        results[0] += 1
      else:
        results[1] += 1

  print(results)

def pg_infersent():
  
  inf = init_infersent()
  object_methods = [method_name for method_name in dir(inf)
                  if callable(getattr(inf, method_name))]


  dp = DataProcessor(data_set="squad")
  dp.load()

  sentences = []
  for i in dp.articles[:10]:
    for j in i.sentences:
      sentences.append(j.text)

  D =inf.get_word_dict(sentences)
  print(D)
  
def pg_squad():
  with open("data/train-v2.0.json") as f:
    data = json.load(f)

  # pprint(data["data"][0]["paragraphs"][0])

  for subject in data["data"]:
    for para in subject["paragraphs"]:
      pprint(para)

def pg_ngrams():
  dp = DataProcessor(data_set="squad")
  dp.load()
  text = dp.articles[0].text

  sentence0 = 'this is a foo bar sentences and i want to ngramize it'
  sentence1 = 'this is also a foo bar sentence, which i want to match'

  n = 2
  bigrams0 = list(bigrams(word_tokenize(sentence0)))
  bigrams1 = list(bigrams(word_tokenize(sentence1)))

  for i in bigrams0:
    for j in bigrams1:
      if i == j:
        print(i)
  print('--')
  for i in word_tokenize(sentence0):
    for j in word_tokenize(sentence1):
      if i == j:
        print(i)


if __name__ == '__main__':
  # pg_squad()
  # pg_infersent_acc()
  # pg_infersent()
  pg_ngrams()


