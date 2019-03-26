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
    self.n_sentences = 0
    self.n_questions = 0
    self.vectors = []
    self.targets = []
    self.n_vectors = 0
    self.word2vec = KeyedVectors.load("data/wordvectors.kv", mmap='r')
    self.n_features = 4
    self.current_art_idx = 0
    self.f_idxs = {
      "cos": 0,
      "unigram": 1,
      "bigram": 2,
      "root_sim": 3
    }

  def load_articles(self, articles):
    self.articles = articles
    self.n_articles = len(self.articles)

    for art in dp.articles:
      self.n_sentences += len(art["sentences"])
      self.n_questions += len(art["questions"])

    self.n_vectors = self.n_sentences * self.n_questions
  
  def create_vectors(self):
    self.vectors = np.zeros((self.n_vectors, self.n_features))
    self.targets = np.zeros(self.n_vectors)
    
    # self.add_cos_scores()
    # self.add_matching_ngrams()
    # self.add_root_sim()
    # self.add_targets()

    vec_idx = 0

    for art in self.articles:
      for sent in art["sentences"]:
        for q_idx, question in enumerate(art["questions"]):
          
          # Cos scores
          self.vectors[vec_idx, self.f_idxs["cos"]] = sent["cos_scores"][q_idx]

          # n-grams
          self.vectors[vec_idx, self.f_idxs["unigram"]] = self.matching_ngrams(
            sent["tokens"], question["question"]["tokens"]
          )

          self.vectors[vec_idx, self.f_idxs["bigram"]] = self.matching_ngrams(
            sent["bigrams"], question["question"]["bigrams"]
          )

          # root sim
          self.vectors[vec_idx, self.f_idxs["root_sim"]] = self.root_sim(
            sent, question["question"]
          )

          vec_idx += 1

  
  # def add_cos_scores(self):
  #   vec_idx = 0

  #   for art in self.articles:
  #     for sent in art["sentences"]:
  #       for cos_score in sent["cos_scores"]:
  #         self.vectors[vec_idx, self.f_idxs["cos"]] = cos_score
  #         vec_idx += 1

  # def add_matching_ngrams(self):
  #   vec_idx = 0

  #   for art in self.articles:
  #     for sent in art["sentences"]:
  #       for question in art["questions"]:

  #         self.vectors[vec_idx, self.f_idxs["unigram"]] = self.matching_ngrams(
  #           sent["tokens"], question["question"]["tokens"]
  #         )

  #         self.vectors[vec_idx, self.f_idxs["bigram"]] = self.matching_ngrams(
  #           sent["bigrams"], question["question"]["bigrams"]
  #         )
  #         vec_idx += 1

  def matching_ngrams(self, ngrams_1, ngrams_2):
    tot = 0
    for i in ngrams_1:
      if i in ngrams_2:
        tot += 1
    return tot
  
  # def add_root_sim(self):
  #   vec_idx = 0

  #   for art in self.articles:
  #     for sent in art["sentences"]:
  #       for question in art["questions"]:
  #         self.vectors[vec_idx, self.f_idxs["root_sim"]] = self.root_sim(
  #           sent, question
  #         )

  def root_sim(self, text_1, text_2):
    tokens_1, tree_1 = text_1["tokens"], text_1["dep_tree"]
    tokens_2, tree_2 = text_2["tokens"], text_2["dep_tree"]

    try:
      return self.word_sim(tokens_1[tree_1[0][2]-1], tokens_2[tree_2[0][2]-1])
    except IndexError:
      return 0

  def word_sim(self, word_1, word_2):
    try:
      return self.word2vec.similarity(word_1, word_2)
    except KeyError:
      return 0

  def add_targets(self):
    v_idx = 0
    for i in range(len(self.article["sentences"])):
      for j in range(len(self.article["questions"])):
        if self.article["questions"][j]["answer"]["answer_sent"] == i:
          self.targets[v_idx] = 1
        else:
          self.targets[v_idx] = 0

        v_idx += 1

if __name__ == "__main__":
  dp = DataProcessor()
  dp.load("data/squad-v6.file")
  
  # av = ArticleVectorizer()
  # av.load_articles(dp.articles)
  # av.create_vectors()

  # print(len(dp.articles[0]["sentences"]))
  # print(len(dp.articles[0]["questions"]))
  # print('---')

  # pprint(dp.articles[0]["sentences"][0]["dep_tree"])
  # print(dp.articles[0]["sentences"][0]["tokens"])
  # pprint(dp.articles[0]["questions"][5]["question"]["dep_tree"])
  # print(dp.articles[0]["questions"][5]["question"]["tokens"])

  # pprint(dp.articles[0]["questions"][3]["question"]["tokens"])
  # print(av.vectors[:20])



