import gensim
from gensim.models import KeyedVectors
import pickle
from pprint import pprint
from data_processing import *
from nltk.tokenize import word_tokenize
import time

if __name__ == "__main__":
  # with open("data/all_squad_sentences.txt", "r") as f:
  #   sentences = [word_tokenize(sent) for sent in f.readlines()]
  
  # model = gensim.models.Word2Vec(sentences)

  # print("Training")
  # model.train(sentences, total_examples=len(sentences), epochs=10)

  # with open("data/w2v.file", "wb") as f:
  #   pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

  # with open("data/w2v.file", "rb") as f:
  #   model = pickle.load(f)



  # s = time.time()
  # model = gensim.models.KeyedVectors.load_word2vec_format(
  #   'data/GoogleNews-vectors-negative300.bin', 
  #   binary=True,
  #   limit=500000)
  # e = time.time()
  # print(e-s)


  # print(model.wv.most_similar(positive="queen"))
  
  # with open('data/google-w2v.file', "wb") as f:
  #   pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

  # with open('data/google-w2v.file', "rb") as f:
  #   model = pickle.load(f)

  # path = "data/wordvectors.kv"
  # model.wv.save(path)

  wv = KeyedVectors.load("data/wordvectors.kv", mmap='r')
  print(wv.most_similar(positive="houston"))