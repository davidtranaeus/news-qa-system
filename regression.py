from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from feature_extraction import Vectorizer
import random
from pprint import pprint
from sklearn.preprocessing import StandardScaler
from time import time
import numpy as np
from data_processing import DataProcessor
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

class LogRegModel():
  def __init__(self):
    self.model = LogisticRegression(solver='lbfgs')
    self.scaler = StandardScaler()

  def load_vectors(self, processed_articles, n_folds=10):
    self.vec = Vectorizer()
    self.vec.vectorize(processed_articles)
    self.vectors = self.vec.vectors
    self.targets = self.vec.targets
    self.vector_ids = self.vec.vector_ids
    kf = KFold(n_splits=n_folds, shuffle=True)
    self.folds = list(kf.split(self.vectors))

  def k_folds(self):
    for fold in self.folds:
      yield fold[0], fold[1]

  def run_k_fold(self, with_sentiment=True):
    print("Running k-fold.")
    X = self.vectors if with_sentiment else self.vectors[:,:4]
    y = self.targets
    
    result = {
      "conf_matrices": [],
      "roc_curves": [],
      "roc_auc_scores": [],
      "coefficients": []
    }

    for train_index, test_index in self.k_folds():
      training_vectors, test_vectors = X[train_index], X[test_index]
      training_targets, test_targets = y[train_index], y[test_index]

      training_vectors = self.scaler.fit_transform(training_vectors)
      test_vectors = self.scaler.transform(test_vectors)

      self.model.fit(
        training_vectors,
        training_targets
      )

      result["conf_matrices"].append(
        confusion_matrix(
          self.model.predict(test_vectors),
          test_targets
      ))

      result["roc_curves"].append(
        roc_curve(
          test_targets, 
          self.model.predict_proba(test_vectors)[:,1])
      )

      result["roc_auc_scores"].append(
        roc_auc_score(
          test_targets, 
          self.model.predict(test_vectors))
      )

      result["coefficients"].append(
        self.model.coef_[0]
      )

    return result
        
if __name__ == "__main__":
  dp = DataProcessor()
  dp.load('data/SQuAD/squad-v7.file')
  model = LogRegModel()
  model.load_vectors(dp.articles)
  model.run_k_fold()