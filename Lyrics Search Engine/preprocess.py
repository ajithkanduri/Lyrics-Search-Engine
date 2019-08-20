import re
import math
import json
import csv
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

def init():
	ps = PorterStemmer()
	tokenizer = RegexpTokenizer('\s+', gaps=True)
	corpus_file = "./dataset/corpus.json"
	docs_file = "./dataset/docs.json"
	songs_data = "./dataset/songdata.csv"
	return ps,tokenizer,corpus_file,docs_file,songs_data

def createDictionary():
	corpus_list = []
	doc_list = {}
	with open(songs_data,'r') as csvfile:
		csvreader = csv.DictReader(csvfile)
		for row in csvreader:
			row_data = []

			raw_artist = str(row["artist"])
			#print(raw_artist)
			#artist = re.sub('[^a-zA-Z]','',raw_artist)
			artist = tokenizer.tokenize(raw_artist)
			artist = list(artist)
			artist = [x.strip('.,!#$%^&*();:\n\t\\\"?!{}[]<>') for x in artist] 
			artist = [token.lower() for token in artist]
			artist = [ps.stem(w) for w in artist]
			row_data.extend(artist)

			raw_song = str(row["song"])
			#song = re.sub('[^a-zA-Z]','',raw_song)
			song = tokenizer.tokenize(raw_song)
			song = list(song)
			song = [x.strip('.,!#$%^&*();:\n\t\\\"?!{}[]<>') for x in song]
			song = [token.lower() for token in song]
			song = [ps.stem(w) for w in song]
			row_data.extend(song)

			raw_lyrics = str(row["text"])
			#lyrics = re.sub('[^a-zA-Z]','',raw_lyrics)
			lyrics = tokenizer.tokenize(raw_lyrics)
			lyrics = list(lyrics)
			lyrics = [x.strip('.,!#$%^&*();:\n\t\\\"?!{}[]<>') for x in lyrics]
			lyrics = [token.lower() for token in lyrics]
			lyrics = [ps.stem(w) for w in lyrics]
			row_data.extend(lyrics)

			corpus_list.append(row_data)

			doc_list_key = raw_song+raw_artist
			doc_list[doc_list_key] = {'tf-idf' : {},'tokens':{}} 
			doc_list[doc_list_key]['tokens'] = row_data

	return corpus_list,doc_list
def tf(word,doc):
	if doc.count(word) == 0 :
		return 0
	else:
		return 1+math.log(doc.count(word),10)
def idf(word,corpus_list):
	count = 0

	for doc in corpus_list :
		if (doc.count(word)) > 0 :
			count = count+1
	if count == 0:
		return -1
	else:
		return math.log(len(corpus_list)/float(count),10)
def tf_idf(word,docs,corpus_list):
	return (tf(word, docs) * idf(word,corpus_list))	

def generateDocVector():
	for doc in doc_list:
		for token in doc_list[doc]['tokens']:
			print(token)
			doc_list[doc]['tf-idf'][token] = tf_idf(token, doc_list[doc]['tokens'], corpus_list)
	
def normalizeDocVector():
	for doc in doc_list:
		doc_len = 0
		for token in doc_list[doc]['tf-idf']:
			doc_len += math.pow((doc_list[doc]['tf-idf'][token]),2) 
		for token in doc_list[doc]['tokens']:
			doc_list[doc]['tf-idf'][token] /= math.sqrt(doc_len)

def writeToFiles(doc_list,corpus_list):
	with open(docs_file, 'w') as json_writer:
	    json.dump(doc_list, json_writer)

	with open(corpus_file, "w") as output:
		json.dump(corpus_list,output)

if __name__ == "__main__":
	ps,tokenizer,corpus_file,docs_file,songs_data = init()
	corpus_list,doc_list = createDictionary()
	#print(corpus_list)
	generateDocVector()
	print("Generated!!")
	normalizeDocVector()
	print("Normalized!")
	writeToFiles(doc_list,corpus_list)
	print("Written")