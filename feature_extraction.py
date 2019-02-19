from data_processing import *

if __name__ == "__main__":
  dp = DataProcessor()
  # dp.process_articles(save=True)
  dp.load()
  articles = dp.articles
  

  for i in articles[0].text:
    print(i)