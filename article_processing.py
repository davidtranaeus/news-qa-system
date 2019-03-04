from textblob import TextBlob
from corenlp import StanfordNLP
from nltk import Tree
import pickle
import json
import re
from pprint import pprint
from nltk import sent_tokenize

class Article():
  def __init__(self, json_data, tokenizer):
    self.type = json_data["type"]
    self.text = re.sub(r'\n\s*\n', '\n\n', json_data["text"])
    # self.sentences = [Sentence(i.__str__(), tokenizer) for i in TextBlob(self.text).sentences]
    self.sentences = [Sentence(i, tokenizer) for i in sent_tokenize(self.text)]
    self.questions = [Question(q_map, self.text, tokenizer) for q_map in json_data["questions"] if "s" in q_map["consensus"]]

class Sentence():
  def __init__(self, text, tokenizer):
    self.text = text
    self.tokens = tokenizer.word_tokenize(text)

  def __repr__(self):
    return self.text

class Question():
  def __init__(self, question_map, text, tokenizer):
    self.span = question_map["consensus"]
    self.q = question_map["q"]
    self.a = text[self.span["s"]:self.span["e"]-1]
    self.q_tokens = tokenizer.word_tokenize(self.q)
    self.a_tokens = tokenizer.word_tokenize(self.a)
    self.find_answer_sentence(text)

  def find_answer_sentence(self, text):
    self.a_sentence_idx = None
    total_len = 0
    for idx, sentence in enumerate(TextBlob(text).sentences):
      # if self.a in sentence:
      #   self.a_sentence_idx = idx

      if self.span["s"] >= total_len and self.span["s"] <= total_len+len(sentence)+1:
        self.a_sentence_idx = idx
        break
      else:
        total_len += len(sentence)+1

    if self.a_sentence_idx == None:
      print("Answer sentence not found")
      print(self.a)
      print(self.span)
      print([i.__str__() for i in TextBlob(text).sentences])
      

class ArticleProcessor():
  def __init__(self, data_path='data/combined-newsqa-data-v1.json'):
    self.data_path = data_path
    self.tokenizer = StanfordNLP()

  def process_data(self, save=False):
    with open(self.data_path) as f:
      data = json.load(f)

    self.n_articles = len(data["data"])
    self.articles = [None for i in range(self.n_articles)]
    
    print("Processing articles ({})".format(self.n_articles))
    for idx in range(self.n_articles):
      # if idx % 1000 == 0:
      print("{} articles processed".format(idx))
      self.articles[idx] = Article(data["data"][idx], self.tokenizer)
    
    if save:
      self.save()

  def save(self, path="data/articles.file"):
    with open(path, "wb") as f:
      pickle.dump(self.articles, f, pickle.HIGHEST_PROTOCOL)

  def load(self, path="data/articles.file"):
    with open(path, "rb") as f:
      self.articles = pickle.load(f)


if __name__ == "__main__":

  ap = ArticleProcessor()
  ap.process_data(save=True)
  # ap.load()
  # pprint(vars(ap.articles[0]))