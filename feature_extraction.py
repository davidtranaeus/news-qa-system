from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_processing import *
from stanford_corenlp import StanfordNLP
from InferSent.models import InferSent
import torch
from nltk import Tree
import numpy as np
import os

# tv = TfidfVectorizer()
# self.tfidf_mat = tv.fit_transform([i.raw_article_text for i in self.articles]).toarray() # (n documents, n unique tokens)
# self.id_to_token = np.asarray(tv.get_feature_names())
# self.token_to_id = {token:idx for idx,token in enumerate(self.id_to_token)}

class FeatureExtractor():
  def __init__(self, articles, data_set='squad'):
    self.data_set = data_set
    self.articles = articles
    self.n_articles = len(self.articles)
    self.n_sentences = sum([len(i.sentences) for i in self.articles])
    self.vectors = [None] * self.n_sentences
    self.get_infersent()

  def get_infersent(self):
    path = 'data/infersent-squad.file' if self.data_set == 'squad' else 'data/infersent-newsqa.file'
    
    if os.path.isfile(path):
      with open(path, "rb") as f:
        self.infersent = pickle.load(f)
    else:
      print("Setting upp infersent")
      V = 1
      # MODEL_PATH = 'InferSent/encoder/infersent%s.pkl' % V
      MODEL_PATH = 'InferSent/encoder/infersent1.pickle'
      params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                      'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
      self.infersent = InferSent(params_model)
      self.infersent.load_state_dict(torch.load(MODEL_PATH))

      W2V_PATH = 'InferSent/dataset/GloVe/glove.840B.300d.txt'
      self.infersent.set_w2v_path(W2V_PATH)

      sentences = []
      for i in dp.articles:
        for j in i.sentences:
          sentences.append(j.text)
        for j in i.questions:
          for k in j.a:
            sentences.append(k)
          for k in j.q:
            sentences.append(k)
      
      self.infersent.build_vocab(sentences, tokenize=True)

      with open(path, "wb") as f:
        pickle.dump(self.infersent, f, pickle.HIGHEST_PROTOCOL)

  def sentence_embeddings(self):
    pass


    


if __name__ == "__main__":
  dp = DataProcessor()
  dp.load("data/squad-v2.file")
  fe = FeatureExtractor(dp.articles)

  print(fe.n_articles)
  print(fe.n_sentences)



