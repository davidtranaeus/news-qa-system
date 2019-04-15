from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from data_processing import *
from feature_extraction import *
import random
from pprint import pprint
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from time import time
import matplotlib.pyplot as plt
import numpy as np

random.seed(100)

class RegressionModel():
  def __init__(self):
    self.dp = DataProcessor()
    self.vec = Vectorizer()
    self.model = LogisticRegression(solver='lbfgs')
    self.scaler = StandardScaler()


  def load(self, path):
    self.dp.load(path)
    self.vec.vectorize(self.dp.articles)
    self.vectors = self.vec.vectors
    self.targets = self.vec.targets
    self.vector_ids = self.vec.vector_ids

  def train(self, split=0.8):
    self.set_train_test(split)
    self.normalize()

    print("Training")
    start = time()
    self.model.fit(
      self.train_vectors,
      self.train_targets
    )
    print("Total time:", time()-start, "\n")

  def set_train_test(self, split):
    total_range = list(range(len(self.vec.articles)))
    random.shuffle(total_range)
    split_idx = round(len(total_range) * split)
    self.train_range, self.test_range = total_range[:split_idx], total_range[split_idx:]

    self.train_vectors = self.vectors[np.in1d(self.vector_ids[:,0], self.train_range)]
    self.train_targets = self.targets[np.in1d(self.vector_ids[:,0], self.train_range)]

    self.test_vectors = self.vectors[np.in1d(self.vector_ids[:,0], self.test_range)]
    self.test_targets = self.targets[np.in1d(self.vector_ids[:,0], self.test_range)]
  
  def normalize(self):
    self.train_vectors = self.scaler.fit_transform(self.train_vectors)
    self.test_vectors = self.scaler.transform(self.test_vectors)
    self.normed_vectors = self.scaler.transform(self.vectors)

  def eval(self):
    print("Model score:", self.model.score(
      self.test_vectors,
      self.test_targets
    ))

    conf_matrix = confusion_matrix(
      self.model.predict(self.test_vectors),
      self.test_targets
    )

    # tn fp
    # fn tp
    tn, fp, fn, tp = conf_matrix.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("Confusion matrix:\n", conf_matrix)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", 2 * (precision * recall) / (precision + recall))
    print("Coef:", self.model.coef_)
    print("n iterations:", self.model.n_iter_)

  def eval_answer_ranking(self):
    res = [0,0]

    probs = self.model.predict_proba(self.normed_vectors)

    for a_idx in self.test_range:
      
      n_questions = int(max(self.vector_ids[np.where(self.vector_ids[:,0] == a_idx)][:,1]))
      
      for q_idx in range(n_questions):
        vec_idxs = np.where(np.all(self.vector_ids == [a_idx, q_idx], axis=1))[0]
        q_probs = probs[vec_idxs]
        targets = self.targets[vec_idxs]

        if np.argmax(q_probs[:,1]) == np.where(targets == 1)[0][0]:
          res[0] += 1
        else:
          res[1] += 1

    print(res) # [10481, 2990]
        

if __name__ == "__main__":
  rm = RegressionModel()
  rm.load('data/SQuAD/squad-v7.file')
  rm.train()
  rm.eval()
  # rm.eval_answer_ranking()