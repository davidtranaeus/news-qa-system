from data_processing import *
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import numpy as np
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class Vectorizer():
  def __init__(self, with_sentiment=True):
    self.vectors = []
    self.targets = []
    self.n_vectors = 0
    self.word2vec = KeyedVectors.load("data/google-word2vec/wordvectors.kv", mmap='r')
    self.with_sentiment = with_sentiment
    self.f_idxs = {
      "cos": 0,
      "unigram": 1,
      "bigram": 2,
      "root_sim": 3
    }
    self.f_idxs = {**self.f_idxs, **{
      "sent_comp": 4,
      "sent_neg": 5,
      "sent_neu": 6,
      "sent_pos": 7
    }} if with_sentiment else self.f_idxs
    self.stopwords = stopwords.words("english")
    # self.lemma = nltk.wordnet.WordNetLemmatizer()
    self.sentiment = SentimentIntensityAnalyzer()

  def vectorize(self, processed_articles):
    print("Creating vectors.")
    self.articles = processed_articles

    for art in self.articles:
      self.n_vectors += len(art["sentences"]) * len(art["questions"])

    self.vectors = np.zeros((self.n_vectors, len(self.f_idxs)))
    self.vector_ids = np.zeros((self.n_vectors, 2))
    self.targets = np.zeros(self.n_vectors)
    
    vec_idx = 0

    for a_idx, art in enumerate(self.articles):
      for s_idx, sent in enumerate(art["sentences"]):
        for q_idx, question in enumerate(art["questions"]):

          # print(a_idx)
          
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

          # sentiment
          if self.with_sentiment:
            
            self.vectors[vec_idx, self.f_idxs["sent_comp"]] = self.sent_sim(
              sent["sentiment"]["compound"], 
              question["question"]["sentiment"]["compound"]
            )

            self.vectors[vec_idx, self.f_idxs["sent_neg"]] = self.sent_sim(
              sent["sentiment"]["neg"], 
              question["question"]["sentiment"]["neg"]
            )

            self.vectors[vec_idx, self.f_idxs["sent_neu"]] = self.sent_sim(
              sent["sentiment"]["neu"], 
              question["question"]["sentiment"]["neu"]
            )

            self.vectors[vec_idx, self.f_idxs["sent_pos"]] = self.sent_sim(
              sent["sentiment"]["pos"], 
              question["question"]["sentiment"]["pos"]
            )

          # target variable
          self.targets[vec_idx] = 1 if question["answer"]["answer_sent"] == s_idx else 0

          self.vector_ids[vec_idx] = [a_idx, q_idx]

          vec_idx += 1
  
  def ngram_match(self, ngrams_1, ngrams_2):
    return len(set(ngrams_1) & set(ngrams_2))
    # try:
    #   if isinstance(ngrams_1[0], tuple):
    #     for bigram in matches:
    #       for token in bigram:
    #         uncommon_matches

    #     return len([match for match in matches if not any(token in self.stopwords for token in match)])
    #   else:
    #     return len([match for match in matches if match not in self.stopwords])
    # except IndexError:
    #   return 0

    # return len(set(ngrams_1) & set(ngrams_2))

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

  def sent_sim(self, sent_1, sent_2):
    return abs(sent_1 - sent_2)

if __name__ == "__main__":
  dp = DataProcessor()
  dp.load("data/SQuAD/squad-v7.file")
  
  vec = Vectorizer()
  vec.vectorize(dp.articles)



