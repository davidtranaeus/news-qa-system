from data_processing import *
from gensim.models import KeyedVectors
import numpy as np

class Vectorizer():
  def __init__(self):
    self.vectors = []
    self.targets = []
    self.n_vectors = 0
    self.word2vec = KeyedVectors.load("data/google-word2vec/wordvectors.kv", mmap='r')
    self.f_idxs = {
      "cos": 0,
      "unigram": 1,
      "bigram": 2,
      "root_sim": 3
    }

  def vectorize(self, articles):
    self.articles = articles

    for art in self.articles:
      self.n_vectors += len(art["sentences"]) * len(art["questions"])

    self.vectors = np.zeros((self.n_vectors, len(self.f_idxs)))
    self.targets = np.zeros(self.n_vectors)
    
    vec_idx = 0

    for art in self.articles:
      for s_idx, sent in enumerate(art["sentences"]):
        for q_idx, question in enumerate(art["questions"]):
          
          # cos scores
          self.vectors[vec_idx, self.f_idxs["cos"]] = sent["cos_scores"][q_idx]

          # n-grams
          self.vectors[vec_idx, self.f_idxs["unigram"]] = self.ngram_match(
            sent["tokens"], question["question"]["tokens"]
          )

          self.vectors[vec_idx, self.f_idxs["bigram"]] = self.ngram_match(
            sent["bigrams"], question["question"]["bigrams"]
          )

          # root sim
          self.vectors[vec_idx, self.f_idxs["root_sim"]] = self.root_sim(
            sent, question["question"]
          )

          # target variable
          self.targets[vec_idx] = 1 if question["answer"]["answer_sent"] == s_idx else 0

          vec_idx += 1
  
  def ngram_match(self, ngrams_1, ngrams_2):
    # TODO stop list
    return len(set(ngrams_1) & set(ngrams_2))
  
  def root_sim(self, text_1, text_2):
    # TODO root for some question is the wh-word
    tokens_1, tree_1 = text_1["tokens"], text_1["dep_tree"]
    tokens_2, tree_2 = text_2["tokens"], text_2["dep_tree"]

    try:
      return self.word_sim(tokens_1[tree_1[0][2]-1], tokens_2[tree_2[0][2]-1])
    except IndexError:
      return 0

  def word_sim(self, word_1, word_2):
    # TODO stemming?
    try:
      return self.word2vec.similarity(word_1, word_2)
    except KeyError:
      return 0

if __name__ == "__main__":
  dp = DataProcessor()
  dp.load("data/SQuAD/squad-v6.file")
  
  vec = Vectorizer()
  vec.vectorize(dp.articles)



