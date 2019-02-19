from textblob import TextBlob
import pickle
import json

class Article():
  def __init__(self, json_data):
    self.process_article(json_data)

  def process_article(self, json_data):
    self.set_text(json_data)
    self.set_sentences(json_data)
    self.set_questions(json_data)

  def set_text(self, json_data):
    self.text = json_data["text"]
  
  def set_sentences(self, json_data):
    self.sentences = [i.__str__() for i in TextBlob(json_data["text"]).sentences]

  def set_questions(self, json_data):
    self.questions = [Question(q, self.text) for q in json_data["questions"] if "s" in q["consensus"]]

class Question():
  def __init__(self, question_map, text):
    self.process_question(question_map, text)

  def process_question(self, question_map, text):
    self.q = question_map["q"]
    self.span = question_map["consensus"]
    self.a = text[self.span["s"]:self.span["e"]-1]

class DataProcessor():
  def __init__(self, data_path='data/combined-newsqa-data-v1.json'):
    self.data_path = data_path

  def process_articles(self, save=False):
    with open(self.data_path) as f:
      data = json.load(f)

    self.n_articles = len(data["data"])
    self.articles = [None for i in range(self.n_articles)]
    
    print("Processing articles ({})".format(self.n_articles))
    for idx in range(self.n_articles):
      if idx % 1000 == 0:
        print("{} articles processed".format(idx))
      self.articles[idx] = Article(data["data"][idx])
    
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
  # dp.process_articles(save=True)
  dp.load()
