import re
import math
import json
import csv
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import pickle

def init():
	ps = PorterStemmer()
	tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
	return ps,tokenizer

def processQuery():
	query = input("Let's Shazammm ")
	query_list = tokenizer.tokenize(query)
	query_list = [token.lower() for token in query_list]
	query_list = [x.strip('-.,!#$%^&*();:\n\t\\\"?!{}[]<>') for x in query_list]
	query_list = [ps.stem(w) for w in query_list]
	return query_list

def output():
	for i,doc in enumerate(x):
		print(doc[0])
		if i ==10:
			break

if __name__ == '__main__':
	ps,tokenizer = init()
	data = pd.read_csv('n_tf_idf.csv')
	# with open('data.pkl','wb') as f:
	# 	pickle.dump(data,f)
	dict_word_idx = {}
	for i,word in enumerate(data.columns):
		dict_word_idx[word] = i
	while(True):
		query_vector = np.zeros(31922)
		query_list = processQuery()
		print(query_list)
		for word in query_list:
			if word in dict_word_idx.keys():
				query_vector[dict_word_idx[word]] = 1
		
		tf_scores = np.matmul(np.array(data),query_vector.T)
		print(np.argsort(tf_scores)[-10:])
	