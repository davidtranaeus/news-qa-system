from textblob import TextBlob
from corenlp import StanfordNLP
from nltk import Tree
import pickle
import json
import re
from pprint import pprint
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.util import bigrams

class Article():
  def __init__(self, text):
    self.text = text
    self.questions = []
    self.sentences = [Sentence(i) for i in sent_tokenize(self.text)]

class Sentence():
  def __init__(self, text):
    self.text = text
    self.tokens = word_tokenize(self.text)
    self.bigrams = list(bigrams(self.tokens))

  def __repr__(self):
    return self.text

class Question():
  def __init__(self, span, question, answer, answer_sentence):
    self.span = span
    self.q = question
    self.a = answer
    self.q_tokens = word_tokenize(self.q)
    self.a_tokens = word_tokenize(self.a)
    self.q_bigrams = list(bigrams(self.q_tokens))
    self.a_bigrams = list(bigrams(self.a_tokens))
    self.a_sentence = answer_sentence

class DataProcessor():
  def __init__(self, data_set="newsqa"):
    self.data_set = data_set

  def process_data(self, save=False):
    if self.data_set == "newsqa":
      self.process_newsqa()
    elif self.data_set == "squad":
      self.process_squad()

    if save:
      self.save()

  def process_newsqa(self):
    with open('data/combined-newsqa-data-v1.json') as f:
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

  def process_squad(self):
    with open("data/train-v2.0.json") as f:
      data = json.load(f)
    # data["data"][0]["paragraphs"][0]

    self.n_articles = 0
    for subject in data["data"]:
      for para in subject["paragraphs"]:
        self.n_articles += 1
    
    self.articles = [None for i in range(self.n_articles)]


    print("Processing articles ({})".format(self.n_articles))
    current_idx = 0
    # for i in range(1):
    #   subject = data["data"][i]
    for subject in data["data"]:
      for para in subject["paragraphs"]:
        print("{} articles processed".format(current_idx))
        
        self.articles[current_idx] = Article(para["context"])

        for q in para["qas"]:
          if q["is_impossible"]:
            continue
          
          answer = q["answers"][0]["text"]
          start = q["answers"][0]["answer_start"]
          sentences = sent_tokenize(para["context"])
          answer_sentence = None
          total_len = 0
          c = 0

          for sentence in sentences:
            if start <= total_len + len(sentence):
              answer_sentence = c
              break
            else:
              total_len += len(sentence)
              c += 1

          if answer_sentence != None:
            self.articles[current_idx].questions.append(Question(
              None,
              q["question"],
              answer,
              answer_sentence
            ))
          
        current_idx += 1


  def save(self):
    if self.data_set == "newsqa":
      path = "data/articles.file"
    else:
      path = "data/squad.file"

    with open(path, "wb") as f:
      pickle.dump(self.articles, f, pickle.HIGHEST_PROTOCOL)

  def load(self):
    if self.data_set == "newsqa":
      path = "data/articles.file"
    else:
      path = "data/squad.file"

    with open(path, "rb") as f:
      self.articles = pickle.load(f)
      self.n_articles = len(self.articles)


if __name__ == "__main__":

  dp = DataProcessor(data_set="squad")
  # dp.process_data(save=True)
  dp.load()

  print(dp.articles[0].sentences[0].bigrams)

  # DEBUG
  # article = dp.articles[0]
  # for i,j in enumerate(article.sentences):
  #   print(i,j)
  # print("\n")
  # for i in article.questions:
  #   print('--')
  #   print(i.q)
  #   print(i.a)
  #   # print(i.span)
  #   print(i.correct_sentence)

