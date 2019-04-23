from utils.stanford_corenlp import StanfordNLP
import pickle
import json
from pprint import pprint
from nltk import sent_tokenize
from nltk.util import bigrams
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class DataProcessor():

  ### TODO
  # - Systematic way to handle infersent scores
  # - Build a stable set for train and dev files of SQuAD

  def __init__(self):
    self.sNLP = StanfordNLP()
    self.sentAnalyzer = SentimentIntensityAnalyzer()

  def save(self, path):
    with open(path, "wb") as f:
      pickle.dump(self.articles, f, pickle.HIGHEST_PROTOCOL)

  def load(self, path):
    print("Loading articles.")
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
      
  def adjust_data(self):
 
    for art_idx, art in enumerate(self.articles):

      for sent_idx, sent in enumerate(art["sentences"]):
        self.articles[art_idx]["sentences"][sent_idx]["sentiment"] = self.sentAnalyzer.polarity_scores(
          sent["text"]
        )
      
      for q_idx, quest in enumerate(art["questions"]):
        self.articles[art_idx]["questions"][q_idx]["question"]["sentiment"] = self.sentAnalyzer.polarity_scores(
          quest["question"]["text"]
        )



  def create_features(self, text, other={}):
    tokens = self.sNLP.word_tokenize(text)
    return {
      "text": text,
      "tokens": tokens,
      "bigrams": list(bigrams(tokens)),
      "pos_tags": self.sNLP.pos(text),
      "dep_tree": self.sNLP.dependency_parse(text),
      "sentiment": self.sentAnalyzer.polarity_scores(text),
      **other 
    }

  def read_squad(self):
    with open("data/SQuAD/train-v2.0.json") as f:
      data = json.load(f)

    self.n_articles = 0
    for subject in data["data"]:
      for para in subject["paragraphs"]:
        self.n_articles += 1

    print("Processing articles ({})".format(self.n_articles))
    current_idx = 0

    self.articles = [{} for i in range(self.n_articles)]

    # for i in range(1):
      # subject = data["data"][i]
    for subject in data["data"]:
      for para in subject["paragraphs"]:
        print("{} articles processed".format(current_idx))
        
        # Article
        self.articles[current_idx] = {
          "text": para["context"],
          "sentences":[self.create_features(sent) for sent in sent_tokenize(para["context"])],
          "questions": []
        }

        # Only include questions if the answer exists in a sentence
        for idx, question in enumerate(para["qas"]):
          if question["is_impossible"]:
            continue
          
          answer_sent = None
          total_len = 0
          sent_idx = 0
          for sent in self.articles[current_idx]["sentences"]:
            if question["answers"][0]["answer_start"] <= total_len + len(sent["text"]):
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
          
        current_idx += 1

    # Remove articles which don't have any questions
    self.articles = [art for art in self.articles if len(art["questions"]) != 0]

if __name__ == "__main__":

  dp = DataProcessor()
  # dp.read_squad()
  dp.load('data/SQuAD/squad-v7.file')
  # dp.adjust_data()
  # dp.save('data/SQuAD/squad-v7.file')

  # DEBUG
  article = dp.articles[1]
  pprint(article)
  for i,j in enumerate(article["sentences"]):
    print(i, j["text"])
  for q in article["questions"]:
    print(q["question"]["text"])
    print(q["answer"]["answer_sent"], q["answer"]["text"])


