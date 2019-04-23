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

# random.seed(100)

class LogRegModel():
  def __init__(self):
    self.model = LogisticRegression(solver='lbfgs')
    self.scaler = StandardScaler()
    self.train_range = None
    self.test_range = None

  def load_vectors(self, processed_articles, with_sentiment=True):
    self.vec = Vectorizer(with_sentiment)
    self.vec.vectorize(processed_articles)
    self.vectors = self.vec.vectors
    self.targets = self.vec.targets
    self.vector_ids = self.vec.vector_ids

  def train(self, filter_test_ids=None, split=0.8, set_train_test_set=False):

    # Model can be retrained and evaluated on the same train/test
    if self.train_range == None or set_train_test_set:
      self.set_train_and_test_range(split)

    self.set_train_set()
    self.set_test_set(filter_test_ids) # all is kept if None
    self.normalize()

    print("Fitting model.\n")
    self.model.fit(
      self.train_vectors,
      self.train_targets)

  def set_train_and_test_range(self, split):
    total_range = list(range(self.vec.n_vectors))
    random.shuffle(total_range)
    split_idx = round(len(total_range) * split)
    self.train_range = total_range[:split_idx]
    self.test_range = total_range[split_idx:]

  def set_train_set(self):
    self.train_vectors = self.vectors[np.in1d(self.vector_ids[:,0], self.train_range)]
    self.train_targets = self.targets[np.in1d(self.vector_ids[:,0], self.train_range)]
  
  def set_test_set(self, filter_test_ids):
    if filter_test_ids:
      # Only keep vectors which are in test range and which id is in filter_test_ids
      test_range = {
        **{i: True for i in self.test_range},
        **{str(i[0])+", "+str(i[1]): True for i in filter_test_ids}
      }

      test_idxs = np.zeros(len(self.vector_ids))
      for idx, v_id in enumerate(self.vector_ids):
        if idx in test_range and str(int(v_id[0]))+", "+str(int(v_id[1])) in test_range:
          test_idxs[idx] = True
      test_idxs = (test_idxs).astype(bool)
      self.test_vectors = self.vectors[test_idxs]
      self.test_targets = self.targets[test_idxs]
    else:
      self.test_vectors = self.vectors[np.in1d(self.vector_ids[:,0], self.test_range)]
      self.test_targets = self.targets[np.in1d(self.vector_ids[:,0], self.test_range)]
  
  def normalize(self):
    self.train_vectors = self.scaler.fit_transform(self.train_vectors)
    self.test_vectors = self.scaler.transform(self.test_vectors)
    self.normed_vectors = self.scaler.transform(self.vectors)

        
if __name__ == "__main__":
  rm = LogRegModel()
  rm.load('data/SQuAD/squad-v7.file')
  rm.train()
  rm.eval()
  # rm.eval_answer_ranking()