from sklearn.feature_extraction.text import TfidfVectorizer
from data_processing import *
from corenlp import StanfordNLP
from nltk import Tree
import numpy as np

class FeatureExtractor():
  def __init__(self, articles):
    self.articles = articles
    self.sNLP = StanfordNLP()
    self.preprocess()
  
  def preprocess(self):
    tv = TfidfVectorizer()
    self.tfidf_mat = tv.fit_transform([i.raw_article_text for i in self.articles]).toarray() # (n documents, n unique tokens)
    self.id_to_token = np.asarray(tv.get_feature_names())
    self.token_to_id = {token:idx for idx,token in enumerate(self.id_to_token)}

  def tfidf(self, doc_id, token):
    if token.lower() in self.token_to_id:
      return self.tfidf_mat[doc_id, self.token_to_id[token.lower()]]
    else:
      return 0 # Return 0 if token not in tfidf dictionary
  

if __name__ == "__main__":
  dp = DataProcessor()
  dp.load()
  fe = FeatureExtractor(dp.articles)