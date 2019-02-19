from corenlp import StanfordNLP
import json
from pprint import pprint
from textblob import TextBlob
from nltk import Tree


if __name__ == '__main__':
  sNLP = StanfordNLP()

  ## Article
  with open('data/combined-newsqa-data-v1.json') as f:
  	data = json.load(f)
  article_idx = 1
  article = data["data"][article_idx]["text"]
  # article = "I shot an elephant in my pyjamas."

  blob = TextBlob(article)
  sentence_idx = 0
  print("\n--- Senctences")
  for i in blob.sentences:
  	print(i)
  sentence = blob.sentences[sentence_idx].__str__()

  print("\n--- sNLP")
  sNLP_parse = sNLP.parse(sentence)
  print(sNLP_parse)

  print("\n--- TextBlob")
  blob_parse = blob.sentences[sentence_idx].parse()
  print(blob_parse)

  print("\n--- NLTK")
  t = Tree.fromstring(sNLP_parse)
  print(t)

  # print("\n---")
  # for i in t.subtrees():
  # 	print(i.leaves())

  print("\n--- Sentence")
  print(sentence)
