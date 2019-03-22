from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_processing import *
from utils.stanford_corenlp import StanfordNLP
from InferSent.models import InferSent
from gensim.models import KeyedVectors
import torch
from nltk import Tree
import numpy as np
import os

class ArticleVectorizer():
  def __init__(self):
    self.n_dim = 4
    self.n_sentences = 0
    self.n_questions = 0
    self.vectors = np.array([])
    self.targets = np.array([])
    self.n_vectors = len(self.vectors)
    self.word2vec = KeyedVectors.load("data/wordvectors.kv", mmap='r')

  # def load_w2v(self):
  #   with open("data/google-w2v.file", "rb") as f:
  #     self.word2vec = pickle.load(f)

  def load_article(self, article):
    self.article = article
    self.n_sentences = len(article["sentences"])
    self.n_questions = len(article["questions"])
  
  def create_vectors(self):
    self.n_vectors = self.n_sentences*self.n_questions
    self.vectors = np.zeros((self.n_vectors, self.n_dim))
    self.targets = np.zeros(self.n_vectors)
    # print(self.n_vectors)
    # print(self.n_sentences)
    # print(self.n_questions)
    # print("\n")

    self.add_cos_scores()
    self.add_matching_ngrams()
    # self.add_wh_type()
    self.add_root_match()
    self.add_targets()
  
  def add_cos_scores(self):
    #print(self.article["sentences"][0]["cos_scores"])
    v_idx = 0
    for i in range(len(self.article["sentences"])):
      for j in range(len(self.article["sentences"][i]["cos_scores"])):
        self.vectors[v_idx,0] = self.article["sentences"][i]["cos_scores"][j]
        v_idx += 1
  
  def add_matching_ngrams(self):
    v_idx = 0
    for i in range(len(self.article["sentences"])):
      for j in range(len(self.article["questions"])):
        self.vectors[v_idx,1] = self.matching_ngrams(
          self.article["sentences"][i]["tokens"],
          self.article["questions"][j]["question"]["tokens"]
        )
        self.vectors[v_idx,2] = self.matching_ngrams(
          self.article["sentences"][i]["bigrams"],
          self.article["questions"][j]["question"]["bigrams"]
        )
        v_idx += 1

  def matching_ngrams(self, ngrams_1, ngrams_2):
    tot = 0
    for i in ngrams_1:
      if i in ngrams_2:
        tot += 1
    return tot
  
  def add_wh_type(self): # varför har jag denna?
    for i in range(len(self.article["questions"])):
      wh_type = self.wh_type(self.article["questions"][i]["question"]["pos_tags"])
      idxs = list(range(i, self.n_vectors, self.n_questions))
      self.vectors[idxs, 4+wh_type] = 1

  def wh_type(self, pos_tags):
    for tag in pos_tags:
      try:
        return ['WRB', 'WP', 'WDT', 'WP$'].index(tag[1])
      except ValueError:
        continue
    return 4

  def add_root_match(self):
    v_idx = 0
    for i in range(len(self.article["sentences"])):
      for j in range(len(self.article["questions"])):
        self.vectors[v_idx,3] = self.root_match(
          self.article["sentences"][i]["dep_tree"],
          self.article["sentences"][i]["tokens"],
          self.article["questions"][j]["question"]["dep_tree"],
          self.article["questions"][j]["question"]["tokens"]
        )
        v_idx += 1

  def root_match(self, tree_1, tokens_1, tree_2, tokens_2):
    try:
      return self.word_sim(tokens_1[tree_1[0][2]-1], tokens_2[tree_2[0][2]-1])
    except IndexError:
      return 0

    # return tokens_1[tree_1[0][2]-1] == tokens_2[tree_2[0][2]-1]

  def word_sim(self, word_1, word_2):
    try:
      return self.word2vec.similarity(word_1, word_2)
    except KeyError:
      return 0

  def add_targets(self):
    v_idx = 0
    for i in range(len(self.article["sentences"])):
      for j in range(len(self.article["sentences"][i]["cos_scores"])):
        if self.article["questions"][j]["answer"]["answer_sent"] == i:
          self.targets[v_idx] = 1
        else:
          self.targets[v_idx] = 0

        v_idx += 1


if __name__ == "__main__":
  dp = DataProcessor()
  dp.load("data/squad-v4.file")
  
  av = ArticleVectorizer()
  av.load_article(dp.articles[0])
  av.create_vectors()



