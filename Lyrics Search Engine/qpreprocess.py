import re
import math
import json
import csv
import copy
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

def init():
	ps = PorterStemmer()
	tokenizer = RegexpTokenizer('\s+', gaps=True)
	songs_data = "./dataset/songdata.csv"
	return ps,tokenizer,songs_data

def createDictionary():
	term_freq = {}
	i=0
	data = pd.read_csv("./dataset/songdata.csv")
	for index,row in 
	
		print("Reading Doc:"+str(i))
		artist = tokenizer.tokenize(row['artist'])
		artist = [x.strip('-.,!#$%^&*();:\n\t\\\"?!{}[]<>') for x in artist] 
		artist = [token.lower() for token in artist]
		artist = [ps.stem(w) for w in artist]

		for w in artist:
			if w not in term_freq.keys():
				term_freq[w] = np.zeros(11094)
			term_freq[w][i] += 1

		raw_song = str(row["song"])
		song = tokenizer.tokenize(raw_song)
		song = [x.strip('-.,!#$%^&*();:\n\t\\\"?!{}[]<>') for x in song]
		song = [token.lower() for token in song]
		song = [ps.stem(w) for w in song]

		for w in song:
			if w not in term_freq.keys():
				term_freq[w] = np.zeros(11094)
			term_freq[w][i] += 1

		raw_lyrics = str(row["text"])
		lyrics = tokenizer.tokenize(raw_lyrics)
		lyrics = [x.strip('-.,!#$%^&*();:\n\t\\\"?!{}[]<>') for x in lyrics]
		lyrics = [token.lower() for token in lyrics]
		lyrics = [ps.stem(w) for w in lyrics]

		for w in lyrics:
			if w not in term_freq.keys():
				term_freq[w] = np.zeros(11094)
			term_freq[w][i] += 1
		i=i+1
	return term_freq
def processQuery()
:	query = input("Let's Shazammm ")
	query_list = tokenizer.tokenize(query)
	query_list = [token.lower() for token in query_list]
	query_list = [x.strip('-.,!#$%^&*();:\n\t\\\"?!{}[]<>') for x in query_list]
	query_list = [ps.stem(w) for w in query_list]
	return query_list
if __name__ == "__main__":
	ps,tokenizer,songs_data = init()
	
	docs_list =[]
	term_freq = createDictionary()
	print("createDictionary")

	tf = pd.DataFrame(term_freq)
	print(tf)
	doc_freq = (tf>0).sum(axis=0)
	inv_doc_freq = np.log10(11094/doc_freq)
	print("idf generated")

	tf = 1 + np.log10(tf)
	tf.replace(to_replace = -np.inf, value = 0, inplace = True)
	print("tf generated")
	print(tf)
	tf_idf = tf.multiply(inv_doc_freq, axis = 1)

	n_tf_idf = tf_idf.multiply(1/(((tf_idf**2).sum(axis=1))**0.5),axis=0)
	print(n_tf_idf)
	dict_word_idx = {}
	for i,word in enumerate(n_tf_idf.columns):
		dict_word_idx[word] = i
	while(True):
		query_vector = np.zeros(31922)
		query_list = processQuery()
		print(query_list)
		for word in query_list:
			if word in dict_word_idx.keys():
				query_vector[dict_word_idx[word]] += 1
		for word in query_list:
			query_vector[dict_word_idx[word]] = query_vector[dict_word_idx[word]]*inv_doc_freq[dict_word_idx[word]]
		tf_scores = np.matmul(np.array(n_tf_idf),query_vector.T)
		print(tf_scores[:10])
		print(np.argsort(tf_scores)[-10:])


