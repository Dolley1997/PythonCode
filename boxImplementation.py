# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:50:39 2020

@author: Dolley
"""

import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
import researchpy as rp
import matplotlib.pyplot as plt
import pingouin as pg
dataset = pd.read_csv('Tag.csv')
q25, q75 = np.percentile(dataset['rssi'], 25), np.percentile(dataset['rssi'], 75)
iqr = q75 - q25
print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
# calculate the outlier cutoff
cut_off = iqr * 1.5
print(cut_off)
lower, upper = q25 - cut_off, q75 + cut_off
# identify outliers
outliers = [x for x in dataset['rssi'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))
# remove outliers
outliers_removed = [x for x in dataset['rssi'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_removed))

#two-way Annova test
#P value of the tag>significant level reject null hypothesis

aov = pg.anova(dv='rssi', between=['Tag', 'Antenna'], data=dataset,
             detailed=True)

print(aov)
#number =LabelEncoder()
#dataset['Tag'] = number.fit_transform(dataset.Tag)
jobs_encoder = LabelBinarizer()
jobs_encoder.fit(dataset['Tag'])
transformed = jobs_encoder.transform(dataset['Tag'])
ohe_df = pd.DataFrame(transformed)
dataset = pd.concat([dataset, ohe_df], axis=1).drop(['Tag'], axis=1)

X = dataset.iloc[:,4:].values
y = dataset.iloc[:,3].values

#X=np.reshape(X, (202004, 1))
#y=np.reshape(y,(202004, 1))
#y=y.astype('int')
# Splitting the dataset into the Training set and Test set  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
"""
# Fitting SVM to the Training set
from sklearn.svm import SVR
classifier = SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix incorrect predictions
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
plt.scatter(X[:, 0], X[:, 1],X[:, 2], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(X_test[:, 0], X_test[:, 1],X_test[:, 2], c=y_test, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim);