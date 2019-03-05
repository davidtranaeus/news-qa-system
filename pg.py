from corenlp import StanfordNLP
import json
from pprint import pprint
from textblob import TextBlob
from nltk import Tree
import spacy
from spacy.tokenizer import Tokenizer
import nltk.data
import re
from nltk import sent_tokenize


if __name__ == '__main__':
  # sNLP = StanfordNLP()

  # ## Article
  with open('data/combined-newsqa-data-v1.json') as f:
  	data = json.load(f)
  article_idx = 0
  article = data["data"][article_idx]["text"]
  questions = data["data"][article_idx]["questions"]
  # article = re.sub(r'\n\s*\n', '\n\n', data["data"][article_idx]["text"])
  
  print(len(article))
  splitted = article.split("\n")
  sentences = [sent_tokenize(i) for i in splitted]
  print(sentences)
  for i in sentences:
    print(len(i))

  span = questions[0]["consensus"]
  # article = " ".join(sent_tokenize(article))
  # print(sent_tokenize(article))
  print(article[span["s"]-6:span["e"]-6])