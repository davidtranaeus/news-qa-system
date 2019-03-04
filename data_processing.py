from textblob import TextBlob
from corenlp import StanfordNLP
from nltk import Tree
import pickle
import json

class Article():
  def __init__(self, json_data, tokenizer):
    self.type = json_data["type"]
    self.process_article(json_data, tokenizer)

  def process_article(self, json_data, tokenizer):
    self.set_text(json_data)
    self.set_sentences(json_data, tokenizer)
    self.set_questions(json_data, tokenizer)

  def set_text(self, json_data):
    self.text = json_data["text"]
  
  def set_sentences(self, json_data, tokenizer):
    self.sentences = [Sentence(i.__str__(), tokenizer) for i in TextBlob(json_data["text"]).sentences]
    
  def set_questions(self, json_data, tokenizer):
    self.questions = [Question(q_map, self.text, tokenizer) for q_map in json_data["questions"] if "s" in q["consensus"]]

class Sentence():
  def __init__(self, text, tokenizer):
    self.text = text
    self.tree = Tree.fromstring(tokenizer.parse(text))

class Question():
  def __init__(self, question_map, text, tokenizer):
    self.process_question(question_map, text, tokenizer)

  def process_question(self, question_map, text, tokenizer):
    self.q = question_map["q"]
    self.span = question_map["consensus"]
    self.a = text[self.span["s"]:self.span["e"]-1]
    self.q_tree = Tree.fromstring(tokenizer.parse(self.q))
    self.a_tree = Tree.fromstring(tokenizer.parse(self.a))
    self.find_q_type()
    # self.q_tokens = tokenizer.word_tokenize(self.q)
    # self.a_tokens = tokenizer.word_tokenize(self.a)
  
  def find_q_type(self):
    self.wh_type = None
    q_types = ["WRB", "WDT", "WP", "WP$"]
    
    for i in self.q_tree.pos():
      if i[1] in q_types:
        pass

    
    # WDT -> which, WP -> who, what, WRB -> where, when, WP$ -> whose

class DataProcessor():
  def __init__(self, data_path='data/combined-newsqa-data-v1.json'):
    self.data_path = data_path
    self.tokenizer = StanfordNLP()

  def process_articles(self, save=False):
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

  dp = DataProcessor()
  # dp.process_articles(save=True)
  dp.load()

  tk = StanfordNLP()

  
  # for i in range(10):
  #   print("\n")
  #   tree_q = Tree.fromstring(tk.parse(dp.articles[i].questions[0].q))
  #   tree_a = Tree.fromstring(tk.parse(dp.articles[i].questions[0].a))
  #   print(tree_q)
  #   print(tree_a)
    # tree.draw()

  tree_q = Tree.fromstring(tk.parse(dp.articles[0].sentences[0]))
  for i in tree_q.subtrees():
    print(i)

  # print(tree.pos()[0][1])
  # tree.draw()