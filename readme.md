This project is (unfortunately) probably not runnable, but it could probgably give some hints of what I was doing during my master's thesis.

# Question answering system
This is the code used for my master's thesis project during the spring of 2019. Still under development.

## Repo structure
data_processing.py - Processes JSON files containing paragraphs, questions, and answers and creates a structured data model.  
feature_extraction.py - Extracts features and creates vector representations used by the ML model.  
regression.py - The setup of the ML model used in this thesis.  
utils - StanfordNLP and word2vec setups used in data_processing.py and feature_extraction.py  
