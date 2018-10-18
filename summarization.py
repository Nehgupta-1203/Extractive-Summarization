from os import listdir
import re
from Stemmer import Stemmer
from nltk.corpus import stopwords
from collections import defaultdict
import math,heapq

#feature tfidf, word in title
def calcTFIDF(vocabWordFreq,totalSentences,sentLabel):
	sentScores = {}	
	for word,data in vocabWordFreq.items():
		idf = 1 + math.log(totalSentences/float(len(data)))
		for sentId,freq in data.items():
			docLen = sentLabel[sentId][0]
			tf = 1 + math.log(freq/float(docLen))
			tf_idf = tf*idf
			if sentId not in sentScores:
				sentScores[sentId] = [tf_idf,1]
			else:
				sentScores[sentId][0] += tf_idf
				sentScores[sentId][1] += 1

	return sentScores

def stemming(data): 
	stemmer=Stemmer("english")
	stemmedData=[stemmer.stemWord(key) for key in data]
	return stemmedData

def removeStopWords(data):
	stop_words = set(stopwords.words('english'))
	filteredData = [word for word in data if not word in stop_words]
	return filteredData

def tokenization(sentence):
	tokenizedData = re.findall(r'[a-z]+|@entity[0-9]+',sentence)
	return tokenizedData

def dataPreprocessing(sentence):
	tokenizedData = tokenization(sentence)
	filteredData = removeStopWords(tokenizedData)
	stemmedData = stemming(filteredData)
	return stemmedData

def calcFeatureFreq(sentFeature,labelData):
	featuretfidfFreq = defaultdict(int)
	featuretitleFreq = defaultdict(int)
	labelFreq = defaultdict(int)

	for feature in sentFeature:
		featuretfidfFreq[feature[0]] += 1
		featuretitleFreq[feature[1]] += 1
	for label in labelData:
		labelFreq[label] += 1
	return featuretitleFreq,featuretfidfFreq,labelFreq


def extractFeature(sentTFIDF,istitle,sentLabel):
	sentFeature = []
	labelData = []	
	for sentId,scores in sentTFIDF.items():
		avgTFIDF = scores[0]/float(scores[1])
		freqTitle = istitle[sentId]
		sLabel = sentLabel[sentId][1]
		sentFeature.append([math.floor(avgTFIDF),freqTitle])
		labelData.append(sLabel)
	return sentFeature,labelData

def createVocab(stemmedData,vocabWordFreq,sentId):
	for word in stemmedData:
		if word not in vocabWordFreq:
			vocabWordFreq[word] = {sentId:1}
		elif sentId not in vocabWordFreq[word]:
			vocabWordFreq[word][sentId] = 1
		else:
			vocabWordFreq[word][sentId] += 1
	return vocabWordFreq

def splitSentLabel(data,title):
	sentLabel = {}
	vocabWordFreq = {}
	allSentences = re.split(r'\n',data)
	totalSentences = len(allSentences)
	sentId = 0
	istitle= {}
	for data in allSentences:
		if len(data) > 0:			
			splitData = re.split(r'\t\t\t',data)
			sentLen = len(splitData[0])
			sentence = ("".join(splitData[0])).lower().strip()
			stemmedData = dataPreprocessing(sentence)
			vocabWordFreq = {**createVocab(stemmedData,vocabWordFreq,sentId),**vocabWordFreq}
			sentLabel[sentId] = [sentLen,int(splitData[1])]
			istitle[sentId] = 0
			for data in stemmedData:
				if data in title:
					istitle[sentId] += 1
			sentId += 1
	return vocabWordFreq,istitle,totalSentences,sentLabel


def naiveBayesClassifier(sentFeature,featuretitleFreq,featuretfidfFreq,labelFreq,totalSentences):
	featureProb = {}

	probLabel0 = labelFreq[0]/totalSentences
	probLabel1 = labelFreq[1]/totalSentences
	probLabel2 = labelFreq[2]/totalSentences

	for index in range(len(sentFeature)):
		tfidf = sentFeature[index][0]
		title = sentFeature[index][1]
		probLabel0 *= (featuretfidfFreq[tfidf]/float(labelFreq[0]))*((1+featuretitleFreq[title])/float(labelFreq[0]))
		probLabel1 *= (featuretfidfFreq[tfidf]/float(labelFreq[1]))*((1+featuretitleFreq[title])/float(labelFreq[1]))
		probLabel2 *= (featuretfidfFreq[tfidf]/float(labelFreq[2]))*((1+featuretitleFreq[title])/float(labelFreq[2]))
		outstr = str(tfidf)+"_"+str(title)
		featureProb[outstr] = max(probLabel0,probLabel1,probLabel2)

	return featureProb

def traindata():
	path = "summarizationdataset/dailymail/training/"
	sentFeature = []
	labelData = []
	totalSentences = 0
	for articleName in listdir(path):
		filename = path + articleName
		file = open(filename,encoding='utf-8')
		data = file.read()
		file.close()
		data = re.split(r'\n\n',data)
		articleTitle = str(data[0]).split("/")[-1]
		articleTitle = articleTitle.split("-")
		articleTitle[-1] = articleTitle[-1].split(".")[0]
		title = defaultdict(int)
		for t in articleTitle:
			title[t] = 1
		articleData = data[1]
		vocabWordFreq,istitle,ttotalSentences,sentLabel = splitSentLabel(articleData,title)
		totalSentences += ttotalSentences
		sentTFIDF = calcTFIDF(vocabWordFreq,totalSentences,sentLabel)
		tsentFeature,tlabelData = extractFeature(sentTFIDF,istitle,sentLabel)
		sentFeature.extend(tsentFeature)
		labelData.extend(tlabelData)		
		articleSummary = data[2]
		entityMapping = data[3]	
		break
	featuretitleFreq,featuretfidfFreq,labelFreq = calcFeatureFreq(sentFeature,labelData)
	featureProb = naiveBayesClassifier(sentFeature,featuretitleFreq,featuretfidfFreq,labelFreq,totalSentences)
	return featureProb



def main():
	featureProb = traindata()	
	
	

if __name__=="__main__":
	main()