from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import numpy as np
from data_processing import *
from feature_extraction import *
import random
from pprint import pprint
from sklearn.metrics import confusion_matrix

random.seed(100)

class RegressionModel():
  def __init__(self):
    self.dp = DataProcessor()
    self.vec = Vectorizer()
    self.n_vectors = 0
    self.model = LogisticRegression(solver="lbfgs")
    # self.clf = SGDClassifier(loss='log')

  def load(self, path):
    self.dp.load(path)
    self.vec.vectorize(self.dp.articles)
    self.vectors = self.vec.vectors
    self.n_vectors = len(self.vectors)
    self.targets = self.vec.targets

  def train(self, split=0.8):
    self.set_train_test(split)

    print("Training")
    self.model.fit(
      self.vectors[self.train_range],
      self.targets[self.train_range]
    )

  def set_train_test(self, split):
    total_range = list(range(self.n_vectors))
    random.shuffle(total_range)
    split_idx = round(len(total_range) * split)
    self.train_range, self.test_range = total_range[:split_idx], total_range[split_idx:]

  def eval(self):
    print("Model score:", self.model.score(
      self.vectors[self.test_range],
      self.targets[self.test_range]
    ))

    conf_matrix = confusion_matrix(
      self.model.predict(self.vectors[self.test_range]),
      self.targets[self.test_range]
    )

    tn, fp, fn, tp = conf_matrix.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("Confusion matrix:\n", conf_matrix)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", 2 * (precision * recall) / (precision + recall))
    print("Coef:", self.model.coef_)
    print("n iterations:", self.model.n_iter_)


if __name__ == "__main__":
  rm = RegressionModel()
  rm.load('data/SQuAD/squad-v6.file')
  rm.train()
  rm.eval()