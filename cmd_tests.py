from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import sys
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

if __name__ == "__main__":
  if sys.argv[1] == "w":
    wv = KeyedVectors.load("data/wordvectors.kv", mmap='r')
    print(wv.similarity(sys.argv[2], sys.argv[3]))
  if sys.argv[1] == "s":
    sia = SentimentIntensityAnalyzer()
    print(sia.polarity_scores(sys.argv[2]))