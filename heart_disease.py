#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 14:55:02 2025

@author: jaiagrawal
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engine.outliers import Winsorizer

heart = pd.read_csv(r"/Users/jaiagrawal/Downloads/beginner_datasets/heart_disease.csv")

'''Data Dictionary
age: Patient's age in years (numerical)

sex: Patient's gender (binary: 1 = male, 0 = female)

chest pain type (cp): Type of chest pain (nominal; 4 possible values, often coded as TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)

resting blood pressure (trestbps): Resting blood pressure (numerical; measured in mm Hg)

serum cholestoral (chol): Serum cholesterol in mg/dl (numerical)

fasting blood sugar (fbs): Fasting blood sugar > 120 mg/dl (binary: 1 = true, 0 = false)

resting electrocardiographic results (restecg): Resting ECG results (nominal; values 0, 1, or 2 indicating normal, ST-T abnormality, or left ventricular hypertrophy)

maximum heart rate achieved (thalach): Maximum heart rate achieved during exercise (numerical)

exercise induced angina (exang): Presence of exercise-induced angina (binary: 1 = yes, 0 = no)

oldpeak: ST depression induced by exercise relative to rest (real; represents the deviation in mm)

the slope of the peak exercise ST segment (slope): Slope of the peak exercise ST segment (nominal; often represented as ‘Down’, ‘Flat’, or ‘Up’)

number of major vessels (ca): Number of major vessels (0–3) colored by fluoroscopy (numerical)

thal: Thalassemia status (integer; 0 = normal, 1 = fixed defect, 2 = reversible defect)

target: Diagnosis of heart disease (binary; 0 = no disease, 1 = disease
'''

heart.describe()

heart.info()

heart.columns                                    

heart = heart.rename(columns={'age':'Age', 'sex':'Sex', 'chest pain type':'Cp', 'resting blood pressure':'Rbp','serum cholestoral in mg/dl      ':'Sc', 'fasting blood sugar > 120 mg/dl ':'Fbs','resting electrocardiographic results':'Rer', 'maximum heart rate achieved  ':'Mhr','exercise induced angina    ':'Eia', 'oldpeak ':'Op', 'slope of peak':'Sp','number of major vessels ':'Mv', 'thal':'Thal'})                                    
heart_col = heart.columns
heart.shape                                  

heart.isnull().sum()
heart.duplicated().sum()

plt.figure(figsize=(10,10))
sns.heatmap(heart.corr(), annot=True,cmap='coolwarm')
plt.show()

sns.pairplot(heart)
plt.show()                                   
 

# age
heart.Age.value_counts()                                  
sns.barplot(x=heart.Age.value_counts()[:10].index, y=heart.Age.value_counts()[:10].values)
plt.xlabel('Age')
plt.ylabel('Age Counter')
plt.show()                                    
                         
heart.Age.describe() 

#checking for outliers
for i in heart_col:
    q1 = heart[i].quantile(0.25)
    q2 = heart[i].quantile(0.50) 
    q3 = heart[i].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5*iqr
    higher_bound = q3 + 1.5*iqr

    print(f"Any {i} out of the range from {lower_bound} to {higher_bound} is an outlier.")

outlier_col = []
for i in heart_col:
    q1 = heart[i].quantile(0.25)
    q2 = heart[i].quantile(0.50) 
    q3 = heart[i].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5*iqr
    higher_bound = q3 + 1.5*iqr
    
    outliers = heart[(heart[i] < lower_bound) | (heart[i] > higher_bound)]
    num_outliers = outliers.shape[0]  # Count the number of rows

    print(f"Number of rows with outliers in {i}: {num_outliers}")
    if num_outliers>0:
        outlier_col.append(i)
        
    
winsor = Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=['Cp','Rbp','Sc','Mhr','Op','Mv'])
heart = winsor.fit_transform(heart)

heart.Fbs.var() # near zero variance so we will drop this column
heart = heart.drop('Fbs',axis=1)
outlier_col.remove('Fbs')

g = heart[(heart.Sc == 380.5)].index
for i in g:
    heart.drop(i,axis=0,inplace=True)


for i in outlier_col:
    q1 = heart[i].quantile(0.25)
    q2 = heart[i].quantile(0.50) 
    q3 = heart[i].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5*iqr
    higher_bound = q3 + 1.5*iqr
    
    outliers = heart[(heart[i] < lower_bound) | (heart[i] > higher_bound)]
    num_outliers = outliers.shape[0]  # Count the number of rows

    print(f"Number of rows with outliers in {i}: {num_outliers}")
    
