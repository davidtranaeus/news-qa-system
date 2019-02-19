'''
A sample code usage of the python package stanfordcorenlp to access a Stanford CoreNLP server.
Written as part of the blog post: https://www.khalidalnajjar.com/how-to-setup-and-use-stanford-corenlp-server-with-python/ 
'''

from stanfordcorenlp import StanfordCoreNLP
import logging
import json
from pprint import pprint
from textblob import TextBlob
from nltk import Tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000

class StanfordNLP:
	def __init__(self, host='http://localhost', port=9000):
		self.nlp = StanfordCoreNLP(host, port=port, timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
		self.props = {
				'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
				'pipelineLanguage': 'en',
				'outputFormat': 'json'
		}

	def word_tokenize(self, sentence):
		return self.nlp.word_tokenize(sentence)

	def pos(self, sentence):
		return self.nlp.pos_tag(sentence)

	def ner(self, sentence):
		return self.nlp.ner(sentence)

	def parse(self, sentence):
		return self.nlp.parse(sentence)

	def dependency_parse(self, sentence):
		return self.nlp.dependency_parse(sentence)

	def annotate(self, sentence):
		return json.loads(self.nlp.annotate(sentence, properties=self.props))

	@staticmethod
	def tokens_to_dict(_tokens):
		tokens = defaultdict(dict)
		for token in _tokens:
			tokens[int(token['index'])] = {
				'word': token['word'],
				'lemma': token['lemma'],
				'pos': token['pos'],
				'ner': token['ner']
			}
			return tokens

if __name__ == '__main__':
	# sNLP = StanfordNLP()

	# ## Article
	# with open('data/combined-newsqa-data-v1.json') as f:
	# 	data = json.load(f)
	# article_idx = 1
	# article = data["data"][article_idx]["text"]
	# # article = "I shot an elephant in my pyjamas."

	# blob = TextBlob(article)
	# sentence_idx = 0
	# print("\n--- Senctences")
	# for i in blob.sentences:
	# 	print(i)
	# sentence = blob.sentences[sentence_idx].__str__()

	# print("\n--- sNLP")
	# sNLP_parse = sNLP.parse(sentence)
	# print(sNLP_parse)

	# print("\n--- TextBlob")
	# blob_parse = blob.sentences[sentence_idx].parse()
	# print(blob_parse)

	# print("\n--- NLTK")
	# t = Tree.fromstring(sNLP_parse)
	# print(t)

	# # print("\n---")
	# # for i in t.subtrees():
	# # 	print(i.leaves())

	# print("\n--- Sentence")
	# print(sentence)
	
	# ## Questions
	# print("\n--- Question")
	# question_idx = 3
	# span = data["data"][article_idx]["questions"][question_idx]["consensus"]
	# pprint(data["data"][article_idx]["questions"][question_idx])
	# print("--- Answer: >{}<".format(data["data"][article_idx]["text"][span["s"]:span["e"]-1]))

	
	with open('data/combined-newsqa-data-v1.json') as f:
		data = json.load(f)

	article_idx = 1
	article = data["data"][article_idx]["text"]
	blob = TextBlob(article)
	# print(article)
	a = "cat hat bat splat cat bat hat mat cat"
	b = "cat mat cat sat"
	c = "I shot an elephant in my pyjamas."
	# tv = CountVectorizer()
	tv = TfidfVectorizer()
	# X = tv.fit_transform([i.__str__() for i in blob.sentences])
	X = tv.fit_transform([a,b,c])
	print(X.toarray())
	print(tv.get_feature_names())