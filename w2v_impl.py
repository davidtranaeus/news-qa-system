import gensim
import pickle
from pprint import pprint
from data_processing import *
from nltk.tokenize import word_tokenize

if __name__ == "__main__":
  # with open("data/all_squad_sentences.txt", "r") as f:
  #   sentences = [word_tokenize(sent) for sent in f.readlines()]
  
  # model = gensim.models.Word2Vec(sentences)

  # print("Training")
  # model.train(sentences, total_examples=len(sentences), epochs=10)

  # with open("data/w2v.file", "wb") as f:
  #   pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

  with open("data/w2v.file", "rb") as f:
    model = pickle.load(f)

  # print(model.wv.vocab.keys())
  # print(model.wv.most_similar(positive="queen"))