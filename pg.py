from utils.stanford_corenlp import StanfordNLP
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
from nltk import pos_tag

# tv = TfidfVectorizer()
# self.tfidf_mat = tv.fit_transform([i.raw_article_text for i in self.articles]).toarray() # (n documents, n unique tokens)
# self.id_to_token = np.asarray(tv.get_feature_names())
# self.token_to_id = {token:idx for idx,token in enumerate(self.id_to_token)}

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
  dp = DataProcessor()
  dp.load("data/squad-v3.file")

  with open("data/cosine_scores.file", "rb") as f:
    scores = pickle.load(f)
  
  res = [0,0]

  print(len(scores))
  print(len(dp.articles))

  for score, art in zip(scores, dp.articles):

    for s, q in zip(score.values(), art["questions"]):

      if np.argmax(s) == q["answer"]["answer_sent"]:
        res[0] += 1
      else:
        res[1] += 1

  print(res)
  

if __name__ == '__main__':
  # pg_squad()
  # pg_infersent_acc()
  # pg_infersent()
  # pg_ngrams()
  # pg_question_type()
  pg_infersent_acc()
  


