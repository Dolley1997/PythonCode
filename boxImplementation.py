# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:50:39 2020

@author: Dolley
"""

import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
dataset = pd.read_csv('Tag.csv')
X = dataset.iloc[:, [1, 3]].values
y = dataset.iloc[:, 4].values  

"""

number =LabelEncoder()
dataset['Tag'] = number.fit_transform(dataset.Tag)
jobs_encoder = LabelBinarizer()
jobs_encoder.fit(dataset['Tag'])
transformed = jobs_encoder.transform(dataset['Tag'])
ohe_df = pd.DataFrame(transformed)
dataset = pd.concat([dataset, ohe_df], axis=1).drop(['Tag'], axis=1)
"""