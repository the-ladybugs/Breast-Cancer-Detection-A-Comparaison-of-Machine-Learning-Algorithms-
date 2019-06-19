# ========================================================================================
#						Cancer Classification - My KNN -	
# ===================================== librairies ===================================

import numpy as np
from collections import Counter
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

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

# ===================================== My KNN ===================================
def train(X_train, y_train):
	# KNN doesn't need training 
	return

def predict(X_train, y_train, x_test, k):
	# create list for distances and targets
	distances = []
	targets = []

	for i in range(len(X_train)):
		# Euclidean Distance
		distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
		# Add to Distances List
		distances.append([distance, i])

	# Sort Distance List
	distances = sorted(distances)

	# Make a list of the k neighbors' targets
	for i in range(k):
		index = distances[i][1]
		targets.append(y_train[index])

	# Return 3 closest classes
	return Counter(targets).most_common(1)[0][0]

def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
	if k > len(X_train):
		print('Can\'t have more neighbors than training samples!!')
	
	# Training Data
	train(X_train, y_train)

	# Prediction
	for i in range(len(X_test)):
		predictions.append(predict(X_train, y_train, X_test[i, :], k))


# =====================================   Data  ==============================================
    
#Data Import
names = ['id','clump_thickness','uniform_cell_size','uniform_cell_shape','marginal_adhesion','single_epi_cell_size','bare_nuclei','bland_chromation','normal_nucleoli','mitoses','class']
df = pd.read_csv('breast-cancer-wisconsin.data.txt',names=names) 

# replace & drop missing values
df= df.replace('?',-1)
df= df.replace(-1,np.nan)
df = df.dropna(how='any') 

# remove id column
df.drop(['id'], 1, inplace=True) 
df = df.astype(float)

#The features X are everything except for the class.
X = np.array(df.drop(['class'], 1))  

# Y is just the class or the diagnosis column 
y = np.array(df['class']) 

# Divise into train and test samples using model selection with ration 0.2
X_train, X_test, y_train, y_test =  model_selection.train_test_split(X, y, test_size=0.2) 

# =====================================   Classification  ==========================================

predictions = []
start = time.time() 
kNearestNeighbor(X_train, y_train, X_test, predictions, 3)
end = time.time()  
predictions = np.asarray(predictions)
time1 = end - start

print("=====================================   Accuracy  ==========================================\n")	

accuracy = accuracy_score(y_test, predictions)
print('Accuracy ', accuracy)
print()

print("========================  Cross validation & Running Time  ===============================\n")
print('Cross Validation', cross(X,y))
print("Training Time ", end - start) 


