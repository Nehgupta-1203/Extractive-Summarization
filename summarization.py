from os import listdir
import re
from Stemmer import Stemmer
from nltk.corpus import stopwords
from collections import defaultdict
import math,heapq
import numpy as np
##features used tfidf,title,uppercasewords,sentpos,sentlen

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

def mapEntity(sentence,entityMapping):
	modSentence = ""
	countUppercase = 0
	words = re.split(r'\s',sentence)
	for word in words:
		if word.islower()==False:
			countUppercase += 1
		if word in entityMapping:
			modSentence += entityMapping[word] + " "
		else:
			modSentence += word + " "
	return modSentence,countUppercase

def entityMap(i,entityMapping):
	entityMap = {}
	allentity = entityMapping.split("\n")
	
	for entity in allentity:
		if len(entity) > 0 and ":" in entity :
			entity = re.split(r':',entity)
			entityMap[entity[0]] = entity[1]
	return entityMap

def createVocab(stemmedData,vocabWordFreq,sentId):
	for word in stemmedData:
		if word not in vocabWordFreq:
			vocabWordFreq[word] = {sentId:1}
		elif sentId not in vocabWordFreq[word]:
			vocabWordFreq[word][sentId] = 1
		else:
			vocabWordFreq[word][sentId] += 1	


def splitSentLabel(data,title,entityMapping):
	sentData = {}
	vocabWordFreq = {}
	allSentences = re.split(r'\n',data)
	sentId = 0
	istitle= {}
	totalSentences = len(allSentences)
	title = dataPreprocessing(title)
	for index,data in enumerate(allSentences):
		if len(data) > 0 :			
			splitData = re.split(r'\t\t\t',data)
			sentence = ("".join(splitData[0])).strip()
			sentence,countUppercase = mapEntity(sentence,entityMapping)
			stemmedData = dataPreprocessing(sentence.lower())
			createVocab(stemmedData,vocabWordFreq,sentId)
			if index < int(totalSentences*0.2) or (totalSentences-index) < int(totalSentences*0.2):
				sentPos = 1
			else:
				sentPos = 0
			sentData[sentId] = [len(sentence),int(splitData[1]),countUppercase,sentPos]
			istitle[sentId] = 0
			for data in stemmedData:
				if data in title:
					istitle[sentId] += 1
			sentId += 1
	return vocabWordFreq,istitle,sentData,totalSentences

def calcTFIDF(vocabWordFreq,totalSentences):
	sentScores = {}	
	for word,data in vocabWordFreq.items():
		idf =  math.log(totalSentences/float(len(data)))
		for sentId,freq in data.items():
			tf = 1 + math.log(freq)
			tf_idf = tf*idf
			if sentId not in sentScores:
				sentScores[sentId] = [tf_idf,1]
			else:
				sentScores[sentId][0] += tf_idf
				sentScores[sentId][1] += 1

	return sentScores

def extractFeature(sentTFIDF,istitle,sentData,inputFeature,target):
	for sentId,scores in sentTFIDF.items():
		avgTFIDF = math.floor(scores[0]/float(scores[1]))
		freqTitle = istitle[sentId]
		sLabel = sentData[sentId][1]
		sentLen = sentData[sentId][0]
		countUppercase = sentData[sentId][2]
		sentPos = sentData[sentId][3]
		if sentLen > 3 and sentLen < 15:
			sLen = 1
		else:
			sLen = 0

		inputFeature.append([avgTFIDF,freqTitle,sLen,countUppercase,sentPos])
		target.append(sLabel)

def evaluate(weights,testdata,actualLabel):
    actualLabel = np.array(actualLabel)
    scores = np.dot(testdata, weights.T)
    scores = np.round(sigmoid(scores))
    predictions = scores.argmax(axis=1)
    print ('Accuracy: {0}'.format((predictions == actualLabel).sum()/float(len(predictions))))

def costFunction(features, target, weights):
    cost = np.array([],dtype=np.float128)
    scores = np.array([],dtype=np.float128)
    scores = np.dot(features, weights)
    cost = np.sum(-( target*scores - np.log(1 + np.exp(scores))))
    return cost

def sigmoid(scores):
    return 1/(1 + np.exp(-scores))

def Logistic(inputFeature,target,num_epochs,learning_rate):
    weights = np.zeros(inputFeature.shape[1],dtype=np.float128)
    for epoch in range(num_epochs):
        scores = np.dot(inputFeature,weights)
        predictions = sigmoid(scores)
        output_error = target - predictions
        gradient = np.dot(inputFeature.T,output_error)
        weights += (learning_rate * gradient)
        if epoch % 10 == 0:
            cost = costFunction(inputFeature, target, weights)
            print("costFunction: ",cost)
            
    return weights

def Multi_Logistic(inputFeature,target,num_epochs,learning_rate):
    target = np.array(target)
    inputFeature = np.array(inputFeature)
    num_classes = 3
    num_feature = inputFeature.shape[1]
    computedClassifier = np.zeros(shape=(num_classes,num_feature),dtype=np.float128)
    for curClass in range(0,num_classes):
        print("Training for label: ",curClass)
        targetLabel =  (target==curClass).astype(int)
        computedClassifier[curClass,:] = Logistic(inputFeature,targetLabel,num_epochs,learning_rate)
    return computedClassifier

def readdata(path):		
	inputFeature = []
	target = []
	for articleName in listdir(path):	
		filename = path + articleName
		file = open(filename,"r",encoding='utf8',errors='ignore')
		data = file.read()
		file.close()
		data = re.split(r'\n\n',data)
		articleTitle = str(data[0]).split("/")[-1]
		articleTitle = articleTitle.split("-")
		articleTitle[-1] = articleTitle[-1].split(".")[0]
		articleTitle = (" ".join(articleTitle))
		articleData = data[1]	
		articleSummary = data[2]
		entityMapping = entityMap(i,data[3])
		vocabWordFreq,istitle,sentData,totalSentences = splitSentLabel(articleData,articleTitle,entityMapping)
		sentTFIDF = calcTFIDF(vocabWordFreq,totalSentences)
		extractFeature(sentTFIDF,istitle,sentData,inputFeature,target)
	return inputFeature,target

def main():
	num_epochs = 100
	learning_rate = 0.1
	print("training")
	trainPath = "summarizationdataset/dailymail/training/"
	testPath = "summarizationdataset/dailymail/test/"
	inputFeature,target = readdata(trainPath)
	classifierWeights = Multi_Logistic(inputFeature,target,num_epochs,learning_rate)
	print("testing")
	testFeature,target = readdata(testPath)
	evaluate(classifierWeights,testFeature,target)
	

if __name__=="__main__":
	main()
