from textblob import TextBlob
from stanford_corenlp import StanfordNLP
from nltk import Tree
import pickle
import json
import re
from pprint import pprint
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from nltk import pos_tag
import os
from InferSent.models import InferSent
from nltk import pos_tag

class Article():
  def __init__(self, text, dep_parser):
    self.text = text
    self.questions = []
    self.sentences = [Sentence(i, dep_parser) for i in sent_tokenize(self.text)]

class Sentence():
  def __init__(self, text, dep_parser):
    self.text = text
    self.tokens = word_tokenize(self.text)
    self.bigrams = list(bigrams(self.tokens))
    self.pos_tags = pos_tag(self.tokens)
    self.dep_tree = dep_parser(self.text)

class Question():
  def __init__(self, span, question, answer, answer_sentence, dep_parser):
    self.span = span
    self.q = question
    self.a = answer
    self.a_sentence = answer_sentence
    self.q_tokens = word_tokenize(self.q)
    self.a_tokens = word_tokenize(self.a)
    self.q_bigrams = list(bigrams(self.q_tokens))
    self.a_bigrams = list(bigrams(self.a_tokens))
    self.q_pos_tags = pos_tag(self.q_tokens)
    self.a_pos_tags = pos_tag(self.a_tokens)
    self.q_dep_tree = dep_parser(self.q)
    self.a_dep_tree = dep_parser(self.a)


class DataProcessor():
  def __init__(self):
    self.articles = None
    self.n_articles = None
    self.sNLP = StanfordNLP()
    
  def process_new_data(self, data_set):
    if data_set == "newsqa":
      self.process_new_newsqa()
    elif data_set == "squad":
      self.process_new_squad()

  def process_existing_data(self):
    assert self.articles != None, "warning: no available articles"

  def process_new_squad(self):
    with open("data/train-v2.0.json") as f:
      data = json.load(f)

    self.n_articles = 0
    for subject in data["data"]:
      for para in subject["paragraphs"]:
        self.n_articles += 1
    
    self.articles = [None for i in range(self.n_articles)]

    print("Processing articles ({})".format(self.n_articles))
    current_idx = 0
    # for i in range(1):
      # subject = data["data"][i]
    for subject in data["data"]:
      for para in subject["paragraphs"]:
        print("{} articles processed".format(current_idx))
        
        self.articles[current_idx] = Article(para["context"], self.sNLP.dependency_parse)

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
              answer_sentence,
              self.sNLP.dependency_parse
            ))
          
        current_idx += 1

  def process_new_newsqa(self):
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

  def save(self, path):
    with open(path, "wb") as f:
      pickle.dump(self.articles, f, pickle.HIGHEST_PROTOCOL)

  def load(self, path):
    # newsqa data/articles.file
    # squad data/squad.file
    with open(path, "rb") as f:
      self.articles = pickle.load(f)
      self.n_articles = len(self.articles)


if __name__ == "__main__":

  dp = DataProcessor()
  # dp.load("data/squad-v2.file")
  # dp.process_existing_data()
  dp.process_new_data("squad")
  # pprint(vars(dp.articles[0]))
  # pprint(vars(dp.articles[10000].questions[0]))
  # pprint(vars(dp.articles[10000].sentences[0]))
  # dp.save('data/squad-v2.file')


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

