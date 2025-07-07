#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 21:03:42 2025

@author: jaiagrawal
"""

import pandas as pd
import numpy as np
df = pd.read_csv(r"bank.csv")

df.duplicated().sum()
df.isna().sum()

df['default'] = df['default'].replace({"no":0})
df['default'] = df['default'].replace({"yes":1})
df['housing'] = df['housing'].replace({"no":0})
df['housing'] = df['housing'].replace({"yes":1})
df['loan'] = df['loan'].replace({"no":0})
df['loan'] = df['loan'].replace({"yes":1})
df['deposit'] = df['deposit'].replace({"no":0})
df['deposit'] = df['deposit'].replace({"yes":1})

df['contact'].value_counts()

df['job'].value_counts()

df['poutcome'].value_counts()

i = df[(df.job=='unknown')].index
df = df.drop(i)


df = df.drop(columns=['marital','education'],axis=1)
df = df.drop(columns=['contact','day','month','poutcome'],axis=1)

X = df.iloc[:,:10]
Y = df.iloc[:,10:]

numeric_features = X.select_dtypes(exclude='object').columns

import matplotlib.pyplot as plt
import seaborn as sns

plt.boxplot(X['age'])
plt.show()
plt.boxplot(X['balance'])
plt.show()
plt.boxplot(X['duration'])
plt.show()

corr = X[numeric_features].corr()
sns.heatmap(corr,cmap="coolwarm",annot=True)
plt.show()

from feature_engine.outliers import Winsorizer
winz = Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['age','balance','duration'])
X = winz.fit_transform(X)

from sklearn.preprocessing import OneHotEncoder
a = OneHotEncoder(sparse_output=False)
X_onehot = a.fit_transform(X[['job']])
X_onehot_enc = pd.DataFrame(X_onehot,columns=a.get_feature_names_out())

X_clean = pd.concat([X.reset_index(drop=True),X_onehot_enc.reset_index(drop=True)],axis=1)
X_clean=X_clean.drop(columns=['job'],axis=1)


from sklearn.preprocessing import StandardScaler
b = StandardScaler()
X_scaled = b.fit_transform(X_clean)
X_scaled_clean = pd.DataFrame(X_scaled,columns=b.get_feature_names_out())

x = X_scaled_clean
Y=Y.reset_index(drop=True)
import statsmodels.api as sm
logit_model = sm.Logit(Y, x).fit()

logit_model.summary()
pred = logit_model.predict(x)

from sklearn import metrics 
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, classification_report

fpr, tpr, thresholds = roc_curve(Y['deposit'], pred)  
optimal_idx = np.argmax(tpr - fpr)  
optimal_threshold = thresholds[optimal_idx]

x["pred"] = np.zeros(len(x))
x.loc[pred > optimal_threshold, "pred"] = 1

confusion_matrix(x.pred, Y['deposit'])  
print('Test accuracy = ', accuracy_score(x.pred, Y['deposit']))
classification = classification_report(x.pred, Y)  
print(classification)
