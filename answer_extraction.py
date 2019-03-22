from data_processing import *
from utils.stanford_corenlp import StanfordNLP
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class AnswerExtractor():
  def __init__(self):
    self.sNLP = StanfordNLP()
    self.wv = KeyedVectors.load("data/wordvectors.kv", mmap='r')

  def get_answer(self, question, sentence):

    try:
      wh_word = self.get_wh_type(question)[0].lower()
    except TypeError:
      wh_word = None

    guess = []

    if wh_word in ['when']:
      guess.extend(self.extract_ent(sentence, "DATE"))

    if wh_word in ["who", "whom"]:
      guess.extend(self.extract_ent(sentence, "PERSON"))

    if wh_word in ["where"]:
      guess.extend(self.extract_ent(sentence, "LOCATION"))

    if wh_word in ["how"]:
      guess.extend(self.extract_how(question, sentence))

    return guess

  def is_noun(self, tag):
    return tag[1] in ["NN", "NNS", "NNP", "NNPS"]

  def is_adj(self, tag):
    return tag[1] in ["JJ", "JJR", "JJS"]

  def get_sub_tree(self, sentence, root_token):
    root_idx = sentence["tokens"].index(root_token)
    paths = [[edge] for edge in sentence["dep_tree"] if edge[1] == root_idx + 1]
    for path in paths:
      for path_edge in path:
        for edge in sentence["dep_tree"]:
          if edge[1] == path_edge[2]:
            path.append(edge)
    return paths

  def w2v_sent(self, tokens):
    vec = np.zeros(300)
    for token in tokens:
      try:
        vec += self.wv[token]
      except KeyError:
        continue
    return vec
  
  def w2v(self, token):
    try:
      return self.wv[token]
    except KeyError:
      return np.zeros(300)

  def get_wh_type(self, question):
    for tag in question["pos_tags"]:
      if tag[1] in ['WRB', 'WP', 'WDT', 'WP$']:
        return tag
        break

  def extract_ent(self, sentence, ent_type):
    return [tag[0] for tag in self.sNLP.ner(sentence["text"]) if tag[1] == ent_type]
  
  def find_similar_token(self, sentence, tokens):
    ent_vector = self.w2v_sent(tokens)
    scores = []
    for token in sentence["tokens"]:
      try:
        scores.append(cosine_similarity([ent_vector], [self.wv[token]]))
      except KeyError:
        scores.append(0)
        continue

    if np.any(scores):
      return sentence["tokens"][np.argmax(np.array(scores).flatten())]
    else:
      for token in tokens: # TODO input token might not exist in sentence
        if token in sentence["text"]:
          return token
      return None

  def get_neighbors(self, root, tree):
    pass
    return [edge for edge in tree if root[1] == edge[1]]

  def extract_how(self, question, sentence):
    dep_tree = question["dep_tree"]
    wh_word = self.get_wh_type(question)
    wh_idx = question["pos_tags"].index(wh_word)
    
    answers = []
    
    path = [edge for edge in dep_tree if edge[2] == wh_idx + 1]

    for edge in dep_tree:
      if edge[2] == path[0][1] and self.is_noun(question["pos_tags"][edge[1] - 1]):
        path.append(edge)

    for path_edge in path[1:]:
      for edge in dep_tree:
        if path_edge[1] == edge[1] and (edge[0] == 'compound' or edge[0] == 'amod'):
          path.append(edge)

    tokens = [question["tokens"][path[0][2] - 1], question["tokens"][path[0][1] - 1]]
    for edge in path[1:]:
      if question["tokens"][edge[1] - 1] not in tokens:
        tokens.append(question["tokens"][edge[1] - 1])
      if question["tokens"][edge[2] - 1] not in tokens:
        tokens.append(question["tokens"][edge[2] - 1])

    print(tokens)    

    if tokens[1] in ["many", "much"]:
      answers = self.extract_ent(sentence, "NUMBER")
      print(answers)

      if tokens[1] == "many":
        answers.extend(self.extract_ent(sentence, "DURATION"))
        answers.extend(self.extract_ent(sentence, "PERCENT"))

      if tokens[1] == "much":
        answers.extend(self.extract_ent(sentence, "MONEY"))

      if len(answers) > 1:
        ent = self.find_similar_token(sentence, tokens[2:])
        print(ent)
        lengths = [abs(sentence["tokens"].index(answer)-sentence["tokens"].index(ent)) for answer in answers]
        print(lengths)
        best_cand_idx = sentence["tokens"].index(answers[np.argmin(lengths)])
        new_answers = [answers[np.argmin(lengths)]]
        for edge in sentence["dep_tree"]:
          if edge[1] == best_cand_idx + 1 and (edge[0] == "nummod" or edge[0] == 'compound'):
            new_answers.append(sentence["tokens"][edge[2] - 1])

            ## Added recently TODO compounds should always be added
            for neigh_edge in sentence["dep_tree"]:
              if neigh_edge[1] == edge[2] and neigh_edge[0] == 'compound':
                new_answers.append(sentence["tokens"][neigh_edge[2] - 1])

          if edge[2] == best_cand_idx + 1 and (edge[0] == "nummod" or edge[0] == 'compound'):
            new_answers.append(sentence["tokens"][edge[1] - 1])

        return [a for a in new_answers if a.lower() not in map(lambda x:x.lower(), question["tokens"])]
          
    return answers




if __name__ == "__main__":
  dp = DataProcessor()
  dp.load('data/squad-v6.file')

  art_id = 2030
  q_id = 1

  art = dp.articles[art_id]

  for idx, q, in enumerate(art["questions"]):
    print(idx, q["question"]["text"])
  print()

  ae = AnswerExtractor()

  for i in range(len(art["questions"])):
  # for i in range(q_id, q_id + 1):
    print(art["questions"][i]["question"]["text"])
    print(art["sentences"][art["questions"][i]["answer"]["answer_sent"]]["text"])
    print(art["questions"][i]["answer"]["text"])

    print(ae.get_answer(
      art["questions"][i]["question"],
      art["sentences"][art["questions"][i]["answer"]["answer_sent"]]
    ))
    print()
