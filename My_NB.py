# ========================================================================================
#						Cancer Classification - My NB -	
# ===================================== librairies ===================================

from collections import Counter
import numpy as np
import pandas as pd
import random
import csv
import math
import time

# ===================================== Cross Validation & Accuracy =======================================

from sklearn  import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

# ===================================== CrossValidation ===================================

def cross(X,y):
	accuracy_total = 0
	for i in range(10):
		predictions = []
		X_train, X_test, y_train, y_test =  model_selection.train_test_split(X, y, test_size=0.2) 
		kNearestNeighbor(X_train, y_train, X_test, predictions, 3)
		predictions = np.asarray(predictions)
		# evaluating accuracy
		accuracy_total += accuracy_score(y_test, predictions)
	cross = accuracy_total / 10
	return cross

# ===================================== My NB ===================================

# Separate into block of 2 classes
def separateByClass(dataset): 
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

# Calculate Mean	
def mean(numbers):
	return sum(numbers)/float(len(numbers))

# Calculate Standard Deviation
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)
	
# Calculate Summary for each Feature
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries
	
# Calculate Summary for each Class
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

# Probability Density of Normal Distribution - Conditional probability of a given attribute value given a class value
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

# Combine Probabilities of Features of a Class by Multiplying Them
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

# Make Predication According to Combined Probabilities : Largest Value		
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

# Make multiple predictions for test data set
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions
	
# Calculate Accuracy
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
# =====================================   Data  ==============================================
#Data Import
names = ['id','clump_thickness','uniform_cell_size','uniform_cell_shape','marginal_adhesion','single_epi_cell_size','bare_nuclei','bland_chromation','normal_nucleoli','mitoses','class']
df = pd.read_csv('breast-cancer-wisconsin.data.txt',names=names) 

# replace & drop missing values
df= df.replace('?',-1)
df= df.replace(-1,np.nan)
df = df.dropna(how='any') 

# remove id column, show that id column is making the accuracy worse
df.drop(['id'], 1, inplace=True) 
df = df.astype(float).values.tolist()

# Divise into train and test samples using model selection with ration 0.2
X_train, X_test =  model_selection.train_test_split(df, test_size=0.2)


# =============================================  Classification ================================================

# Training
start = time.time()
summaries = summarizeByClass(X_train)
end = time.time()

# Testing
start2 = time.time() 
predictions = getPredictions(summaries, X_test)
end2 = time.time()

# Running Time Calculations
time1 = end - start
time2 = end2 - start2

print("=====================================   Accuracy  ==========================================\n")	
accuracy = getAccuracy(X_test, predictions)
print("Accuracy: {0}%".format(accuracy))


print("Training Time ", end - start) 
print("Testing Time ", end2 - start2) 
print("Total Time",time1+time2 )
