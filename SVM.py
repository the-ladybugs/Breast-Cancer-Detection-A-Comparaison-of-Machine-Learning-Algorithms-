# ========================================================================================
#						Cancer Classification - SVM Classifiers -	
# ===================================== librairies =======================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
style.use("ggplot")

# ===================================== Classifiers =======================================

from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC

# ===================================== Cross Validation & Accuracy =======================================

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn  import model_selection, svm ,metrics

# =====================================   Data  ==========================================

names = ['id','clump_thickness','uniform_cell_size','uniform_cell_shape','marginal_adhesion','single_epi_cell_size','bare_nuclei','bland_chromation','normal_nucleoli','mitoses','class']
df = pd.read_csv('breast-cancer-wisconsin.data.txt',names=names) 

# replace missing values
df= df.replace('?',-1)
df= df.replace(-1,np.nan)
df = df.dropna(how='any') 

# remove id column, show that id column is making the accuracy worse
df.drop(['id'], 1, inplace=True) 


#The features X are everything except for the class.
X = np.array(df.drop(['class'], 1))  

# Y is just the class or the diagnosis column
y = np.array(df['class']) 

# Divise into train and test samples using model selection with ration 0.2
X_train, X_test, y_train, y_test =  model_selection.train_test_split(X, y, test_size=0.2) 

# =====================================   Classification  ==========================================

#Define the classifiers:
clf = svm.SVC() 
clf_nu = NuSVC()
clf_lsvc = LinearSVC()

#Train Classifiers
clf.fit(X_train, y_train) 
clf_nu.fit(X_train, y_train)
clf_lsvc.fit(X_train, y_train)

#Accuracy Test
accuracy_svm = clf.score(X_test, y_test) 
accuracy_nu = clf_nu.score(X_test, y_test) 
accuracy_lsvc = clf_lsvc.score(X_test, y_test) 

print("=====================================   Accuracy  ==========================================\n")	

print("SVC", accuracy_svm)
print("NuSVC", accuracy_nu)
print("Linear SVC", accuracy_lsvc)
print()

# Cross Validation
predicted_svm = cross_val_predict(clf, X, y, cv=10)
predicted_nu = cross_val_predict(clf_nu, X, y, cv=10)
predicted_lsvc = cross_val_predict(clf_lsvc, X, y, cv=10)

print("=====================================   10 Fold Crossvalidation  ==========================================\n")
print("SVC Cross-validation",metrics.accuracy_score(y, predicted_svm))
print("NuSVC Cross-validation",metrics.accuracy_score(y, predicted_nu))
print("Linear Cross-validation",metrics.accuracy_score(y, predicted_lsvc))
print()

print("========================  Cross validation: Mean & Standard Deviation & Running Time  ===============================\n")

seed = 7
models = []
models.append(('SVC', svm.SVC()))
models.append(('NuSVC', NuSVC()))
models.append(('LinearSVC', LinearSVC(random_state=0)))
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	start = time.time() 
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	end = time.time()
	results.append(cv_results)
	names.append(name)
	msg = "%s: Mean = %f  Std = %f Running Time = %f" % (name, cv_results.mean(), cv_results.std(), end - start)
	print(msg)
	
print()	
# =====================================  Box Plot Comparaison  ==========================================	
print("========================  Box Plot Comparaison ===============================\n")
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111) 
plt.boxplot(results) #outlier plot knn & give definition of boxplot with meaning
ax.set_xticklabels(names)
plt.show()

