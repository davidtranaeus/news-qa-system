from textblob import TextBlob
from corenlp import StanfordNLP
from nltk import Tree
import pickle
import json
import re
from pprint import pprint
from nltk import sent_tokenize

class Article():
  def __init__(self, text):
    self.text = text
    self.questions = []
    self.sentences = [Sentence(i) for i in sent_tokenize(self.text)]

class Sentence():
  def __init__(self, text):
    self.text = text

  def __repr__(self):
    return self.text

class Question():
  def __init__(self, span, question, answer, correct_sentence):
    self.span = span
    self.q = question
    self.a = answer
    self.correct_sentence = correct_sentence

class DataProcessor():
  def __init__(self, data_path='data/combined-newsqa-data-v1.json'):
    self.data_path = data_path
    # self.tokenizer = StanfordNLP()

  def process_newsqa(self, save=False):
    with open(self.data_path) as f:
      data = json.load(f)

    self.n_articles = len(data["data"])
    self.articles = [None for i in range(self.n_articles)]
    
    print("Processing articles ({})".format(self.n_articles))
    for idx in range(self.n_articles):
    # for idx in range(1):
      print("{} articles processed".format(idx))
      self.articles[idx] = Article(data["data"][idx]["text"])

      for q in data["data"][idx]["questions"]:
        if not "s" in q["consensus"]:
          continue

        start = q["consensus"]["s"]
        end = q["consensus"]["e"]
        answer = data["data"][idx]["text"][start:end]
        split = data["data"][idx]["text"].split("\n")
        
        splitted = [sent_tokenize(i) for i in split]
        # print(splitted)
        answer_sentence = None
        total_len = 0
        c = 0
        for i in splitted:
          if len(i) == 0:
            total_len += 1
          else:
            for j in i:
              if start >= total_len and end < total_len + len(j):
                answer_sentence = c
              
              total_len += len(j)
              c += 1
              
        if answer_sentence != None:
          self.articles[idx].questions.append(Question(
            q["consensus"],
            q["q"],
            answer,
            answer_sentence
          ))
    
    if save:
      self.save()

  def save(self, path="data/articles.file"):
    with open(path, "wb") as f:
      pickle.dump(self.articles, f, pickle.HIGHEST_PROTOCOL)

  def load(self, path="data/articles.file"):
    with open(path, "rb") as f:
      self.articles = pickle.load(f)


if __name__ == "__main__":

  dp = DataProcessor()
  # dp.process_newsqa(save=True)
  dp.load()

  # article = dp.articles[0]
  # for i,j in enumerate(article.sentences):
  #   print(i,j)
  # print("\n")
  # for i in article.questions:
  #   print('--')
  #   print(i.q)
  #   print(i.a)
  #   print(i.span)
  #   print(i.correct_sentence)

