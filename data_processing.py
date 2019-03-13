from utils.stanford_corenlp import StanfordNLP
import pickle
import json
from pprint import pprint
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from nltk import pos_tag
import os
from InferSent.models import InferSent
from nltk import pos_tag

class DataProcessor():
  def __init__(self):
    self.sNLP = StanfordNLP()

  def save(self, path):
    with open(path, "wb") as f:
      pickle.dump(self.articles, f, pickle.HIGHEST_PROTOCOL)

  def load(self, path):
    with open(path, "rb") as f:
      self.articles = pickle.load(f)
      self.n_articles = len(self.articles)
  
  def add_cosine_scores(self, path="data/cosine_scores.file"):
    with open(path, "rb") as f:
      all_scores = pickle.load(f)

    for i in range(len(self.articles)):
      for j in range(len(self.articles[i]["sentences"])):
        self.articles[i]["sentences"][j]["cos_scores"] = []
        for k in range(len(all_scores[i])):
          self.articles[i]["sentences"][j]["cos_scores"].append(all_scores[i][k][j])
      

  def create_features(self, text, other={}):
    tokens = word_tokenize(text)
    return {
      "text": text,
      "tokens": tokens,
      "bigrams": list(bigrams(tokens)),
      "pos_tags": pos_tag(tokens),
      "dep_tree": self.sNLP.dependency_parse(text),
      **other 
    }

  def read_squad(self):
    with open("data/train-v2.0.json") as f:
      data = json.load(f)

    self.n_articles = 0
    for subject in data["data"]:
      for para in subject["paragraphs"]:
        self.n_articles += 1

    self.articles = [{} for i in range(self.n_articles)]

    print("Processing articles ({})".format(self.n_articles))
    current_idx = 0

    # for i in range(1):
    #   subject = data["data"][i]
    for subject in data["data"]:
      for para in subject["paragraphs"]:
        print("{} articles processed".format(current_idx))
        
        # Article
        self.articles[current_idx]["text"] = para["context"]

        # Sentences
        self.articles[current_idx]["sentences"] = [
          self.create_features(sent) for sent in sent_tokenize(para["context"])
        ]

        # Questions
        self.articles[current_idx]["questions"] = []

        # Only include questions if the answer exists in a sentence
        for idx, question in enumerate(para["qas"]):
          if question["is_impossible"]:
            continue
          
          answer_sent = None
          total_len = 0
          sent_idx = 0
          for sent in self.articles[current_idx]["sentences"]:
            if question["answers"][0]["answer_start"] <= total_len + len(sent):
              answer_sent = sent_idx
              break
            else:
              total_len += len(sent["text"])
              sent_idx += 1


          if answer_sent != None:
            self.articles[current_idx]["questions"].append({
              "question": self.create_features(question["question"]),
              "answer": self.create_features(
                question["answers"][0]["text"],
                {"answer_sent": answer_sent}),
            })
          
        if len(self.articles[current_idx]["questions"]) == 0:
          print("No questions", current_idx)
          
        current_idx += 1

  def read_newsqa(self):
    pass
    # with open('data/combined-newsqa-data-v1.json') as f:
    #   data = json.load(f)

    # self.n_articles = len(data["data"])
    # self.articles = [None for i in range(self.n_articles)]
    
    # print("Processing articles ({})".format(self.n_articles))
    # for idx in range(self.n_articles):
    # # for idx in range(1):
    #   print("{} articles processed".format(idx))
    #   self.articles[idx] = Article(data["data"][idx]["text"])

    #   for q in data["data"][idx]["questions"]:
    #     if not "s" in q["consensus"]:
    #       continue

    #     start = q["consensus"]["s"]
    #     end = q["consensus"]["e"]
    #     answer = data["data"][idx]["text"][start:end]
    #     split = data["data"][idx]["text"].split("\n")
        
    #     splitted = [sent_tokenize(i) for i in split]
    #     # print(splitted)
    #     answer_sentence = None
    #     total_len = 0
    #     c = 0
    #     for i in splitted:
    #       if len(i) == 0:
    #         total_len += 1
    #       else:
    #         for j in i:
    #           if start >= total_len and end < total_len + len(j):
    #             answer_sentence = c
              
    #           total_len += len(j)
    #           c += 1
              
    #     if answer_sentence != None:
    #       self.articles[idx].questions.append(Question(
    #         q["consensus"],
    #         q["q"],
    #         answer,
    #         answer_sentence
    #       ))



if __name__ == "__main__":

  dp = DataProcessor()
  dp.load("data/squad-v4.file")
  # dp.add_features()
  pprint(dp.articles[1247])

  # # DEBUG
  # article = dp.articles[10000]
  # for i,j in enumerate(article["sentences"]):
  #   print(i, j["text"])
  # for q in article["questions"]:
  #   print(q["question"]["text"])
  #   print(q["answer"]["answer_sent"], q["answer"]["text"])


