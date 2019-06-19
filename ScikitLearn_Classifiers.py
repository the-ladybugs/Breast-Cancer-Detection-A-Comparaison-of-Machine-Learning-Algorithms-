# ========================================================================================
#						Cancer Classification - ScikitLearn Classifiers -	
# ===================================== librairies =======================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# ===================================== Classifiers =======================================

from sklearn  import model_selection, svm ,metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import VarianceThreshold

# ===================================== Cross Validation & Accuracy =======================================

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

# =====================================   Data  ==========================================
names = ['id','clump_thickness','uniform_cell_size','uniform_cell_shape','marginal_adhesion','single_epi_cell_size','bare_nuclei','bland_chromation','normal_nucleoli','mitoses','class']
df = pd.read_csv('breast-cancer-wisconsin.data.txt',names=names) 


# replace missing values
df= df.replace('?',-1)
df= df.replace(-1,np.nan)
df = df.dropna(how='any') 

# remove id column
df.drop(['id'], 1, inplace=True) 


#The features X are everything except for the class.
X = np.array(df.drop(['class'], 1))  

# Y is just the class or the diagnosis column 
y = np.array(df['class']) 

# Divise into train and test samples using model selection with ration 0.2
X_train, X_test, y_train, y_test =  model_selection.train_test_split(X, y, test_size=0.2) 

# =====================================   Classification  ==========================================

#Define the classifiers:
clf_knn = KNeighborsClassifier()
clf_nb = GaussianNB()
clf_lsvc = LinearSVC(random_state=0)


#Train Classifiers
clf_knn.fit(X_train, y_train)
clf_nb.fit(X_train, y_train)
clf_lsvc.fit(X_train, y_train)

#Accuracy Test
accuracy_nb =  clf_nb.score(X_test, y_test)
accuracy_knn = clf_knn.score(X_test, y_test) 
accuracy_lsvc = clf_lsvc.score(X_test, y_test) 

print("=====================================   Accuracy  ==========================================\n")	
print("NB", accuracy_nb)
print("KNN", accuracy_knn)	
print("Linear SVC", accuracy_lsvc)   
print()

# Cross Validation
predicted_nb = cross_val_predict(clf_nb, X, y, cv=10)
predicted_knn = cross_val_predict(clf_knn, X, y, cv=10)
predicted_lsvm = cross_val_predict(clf_lsvc, X, y, cv=10)

print("=====================================   10 Fold Crossvalidation  ==========================================\n")
print("NB Cross-validation",metrics.accuracy_score(y, predicted_nb))
print("KNN Cross-validation",metrics.accuracy_score(y, predicted_knn))
print("Linear Cross-validation",metrics.accuracy_score(y, predicted_lsvm))
print()


print("========================  Cross validation: Mean & Standard Deviation & Running Time  ===============================\n")

seed = 7
models = []
models.append(('Linear SVC', LinearSVC(random_state=0)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))

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