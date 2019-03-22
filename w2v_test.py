from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import sys

if __name__ == "__main__":
  wv = KeyedVectors.load("data/wordvectors.kv", mmap='r')
  print(wv.similarity(sys.argv[1], sys.argv[2]))
  