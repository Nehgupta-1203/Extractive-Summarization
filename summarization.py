from os import listdir
import re
from Stemmer import Stemmer
from nltk.corpus import stopwords
from collections import defaultdict
import math,heapq
sentFeature = defaultdict(lambda: defaultdict(lambda: int))
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

def extractFeature(sentTFIDF,istitle,sentLabel):
	global sentFeature
	for sentId,scores in sentTFIDF.items():
		avgTFIDF = math.floor(scores[0]/float(scores[1]))
		freqTitle = istitle[sentId]
		sLabel = sentLabel[sentId][1]
		sentFeature[sLabel][avgTFIDF] += 1
		sentFeature[sLabel][freqTitle] += 1 
	
	return sentFeature

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


def calcFeatureProb(totalSentences):
	featureProb = defaultdict(lambda: defaultdict(lambda: int))
	probLabel0 = labelFreq[0]/totalSentences
	probLabel1 = labelFreq[1]/totalSentences
	probLabel2 = labelFreq[2]/totalSentences
	#data:=tfidf,title
	for sLabel,data in sentFeature.items():
		noOfentry =  len(data)/2
		for d,freq in data.items():			
			featureProb[sLabel][d] = freq/float(noOfentry)

	return featureProb,probLabel0,probLabel1,probLabel2

def traindata():
	path = "summarizationdataset/dailymail/training/"
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
		extractFeature(sentTFIDF,istitle,sentLabel)			
		articleSummary = data[2]
		entityMapping = data[3]
	featureProb,probLabel0,probLabel1,probLabel2 = calcFeatureProb(totalSentences)
	return featureProb,probLabel0,probLabel1,probLabel2


accuracy = 0
def checkAccuracy(predictedLabel,sentLabel,totalSentences):
	global accuracy
	count = 0
	for sentId,label in predictedLabel.items():
		if label == sentLabel[sentId]:
			count += 1

	accuracy += count/float(totalSentences)


def naiveBayesClassifier(testSentFeature,featureProb,probLabel0,probLabel1,probLabel2):
	predictedProb = {}
	for sentId,data in testSentFeature.items():
		for d in data:
			probLabel0 =* featureProb[0][d[0]] * featureProb[0][d[1]]
			probLabel1 =* featureProb[1][d[0]] * featureProb[1][d[1]]
			probLabel2 =* featureProb[2][d[0]] * featureProb[2][d[1]]
		predictedProb[sentId] = max(probLabel0,probLabel1,probLabel2)
	return predictedProb

def extractTestFeature(sentTFIDF,istitle):
	sentFeature = defaultdict(lambda: list())
	for sentId,scores in sentTFIDF.items():
		avgTFIDF = math.floor(scores[0]/float(scores[1]))
		freqTitle = istitle[sentId]
		sentFeature[sentId].append([avgTFIDF,freqTitle])		
	return sentFeature

def testdata():
	path = "summarizationdataset/dailymail/testing/"
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
		articleSummary = data[2]
		entityMapping = data[3]	
		
		vocabWordFreq,istitle,totalSentences,sentLabel = splitSentLabel(articleData,title)
		totalSentences += ttotalSentences
		sentTFIDF = calcTFIDF(vocabWordFreq,totalSentences,sentLabel)
		testSentFeature = extractTestFeature(sentTFIDF,istitle)	
		predictedProbLabel = naiveBayesClassifier(testSentFeature,featureProb,probLabel0,probLabel1,probLabel2)
		checkAccuracy(predictedLabel,sentLabel,totalSentences)

def main():
	featureProb,probLabel0,probLabel1,probLabel2 = traindata()
	testdata(featureProb,probLabel0,probLabel1,probLabel2)
	print("Accuracy of Model: ", accuracy)
	

if __name__=="__main__":
	main()
