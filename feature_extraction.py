from sklearn.feature_extraction.text import TfidfVectorizer
from data_processing import *
from corenlp import StanfordNLP
import nltk
import numpy as np

class FeatureExtractor():
  def __init__(self, articles):
    self.articles = articles
    self.sNLP = StanfordNLP()
    self.preprocess()
  
  def preprocess(self):
    tv = TfidfVectorizer()
    self.tfidf = tv.fit_transform([i.text for i in self.articles]).toarray()
    self.id_to_token = np.asarray(tv.get_feature_names())

    self.token_to_idx = dict()
    for i in range(len(self.id_to_token)):
      self.token_to_idx[self.id_to_token[i]] = i

  def tokens(self, span):
    return sNLP.word_tokenize(span)



# def matching_word_frequencies(c, s, q, tfidf, id_to_token, token_to_id):
#   matches = set(s).intersection(q)

#   for i in matches:
#     print(matches)
#     print(tfidf[0][token_to_id[i]])


if __name__ == "__main__":
  dp = DataProcessor()
  # dp.process_articles(save=True)
  dp.load()
  articles = dp.articles
  fe = FeatureExtractor(articles)
  print(fe.tfidf)
