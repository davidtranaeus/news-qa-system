from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import numpy as np
from data_processing import *
from feature_extraction import *
import random
from pprint import pprint
from sklearn.metrics import confusion_matrix

random.seed(100)

class LogisticClassifier():
  def __init__(self):
    self.dp = DataProcessor()
    self.av = ArticleVectorizer()
    self.n_articles = 0
    # self.model = LogisticRegression(solver="lbfgs")
    self.clf = SGDClassifier(loss='log')

  def load_data(self, path):
    self.dp.load(path)
    self.n_articles = self.dp.n_articles

  def load_article(self, article_idx):
    self.av.load_article(self.dp.articles[article_idx])
    self.av.create_vectors()
    self.batch_vectors = self.av.vectors
    self.batch_targets = self.av.targets

  def train(self, split=0.8):
    self.split_train_test(0.8)

    for i in range(1):
      print(i)
      for batch_n, idx in enumerate(self.train_range):
        self.load_article(idx)

        # TODO Some articles does not have any questions
        if len(self.batch_vectors) == 0:
          continue
        
        # Shuffle
        c = list(zip(self.batch_vectors, self.batch_targets))
        random.shuffle(c)
        self.batch_vectors, self.batch_targets = zip(*c)
        self.clf.partial_fit(self.batch_vectors, self.batch_targets, [0,1])

  def split_train_test(self, split):
    total_range = list(range(self.n_articles))
    random.shuffle(total_range)
    self.train_range = total_range[:round(len(total_range)*split)]
    self.test_range = total_range[round(len(total_range)*split):]

  def eval(self):
    acc = 0
    for i in self.test_range:
      self.load_article(i)
      # TODO Some articles does not have any questions
      if len(self.batch_vectors) == 0:
        continue
      acc += self.clf.score(self.batch_vectors, self.batch_targets)
    print(acc/len(self.test_range))
    print(self.clf.coef_)


if __name__ == "__main__":
  lc = LogisticClassifier()
  lc.load_data('data/squad-v4.file')
  lc.train(split=0.8)
  lc.eval()