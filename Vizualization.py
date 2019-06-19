# ========================================================================================
#						Cancer Classification - Dataset Vizualization -	
# ===================================== librairies =======================================
import numpy as np 
import pandas as pd
# data visualization library   
import seaborn as sns 
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score


# =====================================   Data  ==========================================

names = ['id','clump_thickness','uniform_cell_size','uniform_cell_shape','marginal_adhesion','single_epi_cell_size','bare_nuclei','bland_chromation','normal_nucleoli','mitoses','diagnosis']
df = pd.read_csv('breast-cancer-wisconsin.data.txt',names=names) 

# replace missing values
df= df.replace('?',-1)
df= df.replace(-1,np.nan)
df = df.dropna(how='any')
df = df.astype(float)

# y includes our labels and x includes our features
y = df.diagnosis                          # M or B 
list = ['id','diagnosis']
x = df.drop(list,axis = 1 )

# Feature names as a list
# columns gives columns names in data 
col = df.columns       
print("==============================  Features ===========================================")
for i in range (len(names)):
    print(names[i])
print ("=========================================================================")


# =====================================   Dataset Statistic Desc  ==============================================

print(x.describe())

# =====================================   1st Plot : Count Plot  ==============================================

ax = sns.countplot(y,label="Count")       # M = 241, B = 458
B, M = y.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)
print('Total Number : ',M+B)
plt.show()

# =====================================   2nd Plot : Median  ==============================================
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)

# =====================================   3rd Plot : Feature Correlation  1 ==============================================

sns.jointplot(x.loc[:,'mitoses'], x.loc[:,'clump_thickness'], kind="regg", color="#ce1414")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')

# =====================================   4th Plot : Feature Correlation 2 ==============================================

sns.jointplot(x.loc[:,'uniform_cell_size'], x.loc[:,'uniform_cell_shape'], kind="regg", color="#ce1414")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')

# =====================================   5th Plot : Correlation Map ==============================================

f,ax = plt.subplots(figsize=(15, 10))
sns.heatmap(x.corr(), annot=True, linewidths=.8, fmt= '.1f',ax=ax)
plt.show()


# =====================================   6th Plot : Feature Importance ==============================================

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(x_train,y_train)
importances = clr_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print(x_train.columns[indices])
"""plt.figure(0.3, figsize=(8, 6))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()"""