import gensim
from gensim.models import KeyedVectors

if __name__ == "__main__":

  model = gensim.models.KeyedVectors.load_word2vec_format(
    'data/google-word2vec/GoogleNews-vectors-negative300.bin', 
    binary=True,
    limit=500000)

  path = "data/wordvectors.kv"
  model.wv.save(path)

  wv = KeyedVectors.load("data/google-word2vec/wordvectors.kv", mmap='r')