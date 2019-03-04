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

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Sgt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


if __name__ == '__main__':
  # sNLP = StanfordNLP()

  # ## Article
  with open('data/combined-newsqa-data-v1.json') as f:
  	data = json.load(f)
  article_idx = 21
  # article = data["data"][article_idx]["text"]
  article = re.sub(r'\n\s*\n', '\n\n', data["data"][article_idx]["text"])
  blob = TextBlob(article)

  # nlp = spacy.load('en_core_web_sm')
  # nlp = en_core_web_sm.load()
  # doc = nlp(article)

  # # for entity in doc.ents:
  # #   print(entity.text, "--", entity.label_)

  # print(split_into_sentences(article))
  # print(len(split_into_sentences(article)))
  # print([i.__str__() for i in TextBlob(article).sentences])
  # print(len([i.__str__() for i in TextBlob(article).sentences]))
  # print(sent_tokenize(article))
  # print(len(sent_tokenize(article)))

  # print(article.split("\n"))
  print(sent_tokenize(article))