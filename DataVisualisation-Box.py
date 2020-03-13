# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:20:08 2020

@author: Dolley
"""
from pandas import Series
from pandas import DataFrame
from pandas import concat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import vstack
from numpy import savetxt
from numpy import array
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from os import listdir
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
# Seaborn visualization library
import seaborn as sns
from math import sqrt
from pandas import datetime

from keras.models import load_model
# Create the default pairplot

fields = ['Time','Tag','Antenna','rssi']

dataset = pd.read_csv('Tag.csv', parse_dates=['Time'],skipinitialspace=True, usecols=fields)
df=pd.DataFrame(dataset)
#ax.set_xlim(dataset['Time'].min(), dataset['Time'].max())
#df.groupby(['Tag','Antenna'])
#df.stack()

data_to_plot = dataset.iloc[:,:3].values
rssi = dataset.iloc[:, -1].values
i = rssi.reshape((rssi.shape[0], 1))
data_to_plot = np.append(data_to_plot, i, axis=1)
dataFrame = pd.DataFrame()
dataFrame['x'] = data_to_plot[:,0]
dataFrame['tags'] = data_to_plot[:,1]
dataFrame['Antenna'] = data_to_plot[:,2]
dataFrame['rssi']=data_to_plot[:,3]
dataFrame.rssi = dataFrame.rssi.astype(float)
dataFrame.Antenna=dataFrame.Antenna.astype(int)
#dataFrame = dataFrame.pivot_table( index='x', columns='Antenna', values='rssi', aggfunc=np.median)
dataFrame = dataFrame.pivot_table( index='x', columns='tags', values='rssi', aggfunc=np.median)
dataFrame.plot()

#Data visualisation w.r.t to time series
def load_dataset(prefix=''):
    grps_dir, data_dir = prefix+'groups/', prefix+'Movement_Rssi_removedStagnant/'
    # load mapping files
    #dtDir_len=data_dir.l
    paths = pd.read_csv(grps_dir + 'MovementPaths.csv', header=0)
    groups = pd.read_csv(grps_dir + 'DatasetGrp.csv', header=0)
    targets = pd.read_csv(grps_dir + 'TargetValues.csv', header=0)
    # load traces
    sequences = list()
    for name in listdir(data_dir):
        
        filename = data_dir + name
        #print(filename)
        #if filename.startswith('Target'):
        #    continue
        df = pd.read_csv(filename, header=0)
        values = df.values
        sequences.append(values)
    return sequences, targets.values[:,1], groups.values[:,1], paths.values[:,1]


# load dataset
sequences, targets, groups, paths = load_dataset()


class1,class2 = len(targets[targets==-1]), len(targets[targets==1])
print('Class=-1: %d %.3f%%' % (class1, class1/len(targets)*100))
print('Class=+1: %d %.3f%%' % (class2, class2/len(targets)*100))

all_rows = vstack(sequences)
#pyplot.figure()
variables = [0, 1]
for v in variables:
    plt.subplot(len(variables), 1, v+1)
    plt.hist(all_rows[:, v], bins=20)
pyplot.show()
# histogram for trace lengths
trace_lengths = [len(x) for x in sequences]
plt.hist(trace_lengths)
plt.show()

def create_dataset(sequences, targets):
    # create the transformed dataset
    transformed = list()
    n_vars = 1
    n_steps = 270
    # process each trace in turn
    for i in range(len(sequences)):
        seq = sequences[i]
        vector = list()
        # last n observations
        for row in range(0, n_steps+1):
            for col in range(n_vars):
                vector.append(seq[row, col])
        # add output
        vector.append(targets[i])
        # store
        print('Transformed')
        transformed.append(vector)
    # prepare array
    transformed = array(transformed)
    transformed = transformed.astype('float32')
    return transformed

# separate traces
seq1 = [sequences[i] for i in range(len(groups)) if groups[i]==1]
seq2 = [sequences[i] for i in range(len(groups)) if groups[i]==2]
seq3 = [sequences[i] for i in range(len(groups)) if groups[i]==3]
seq4 = [sequences[i] for i in range(len(groups)) if groups[i]==4]
#print(len(seq1),len(seq2),len(seq3))
# separate target
targets1 = [targets[i] for i in range(len(groups)) if groups[i]==1]
targets2 = [targets[i] for i in range(len(groups)) if groups[i]==2]
targets3 = [targets[i] for i in range(len(groups)) if groups[i]==3]
targets4 = [targets[i] for i in range(len(groups)) if groups[i]==4]
#print(len(targets1),len(targets2),len(targets3))

es1 = create_dataset(seq1+seq2+seq3+seq4, targets1+targets2+targets3+targets4)
#print('ES1: %s' % str(es1.shape))
savetxt('es1.csv', es1, delimiter=',')
# create ES2 dataset
es2_train = create_dataset(seq1+seq2, targets1+targets2)
es2_test = create_dataset(seq3+seq4, targets3+targets4)
print('ES2 Train: %s' % str(es2_train.shape))
print('ES2 Test: %s' % str(es2_test.shape))
savetxt('es2_train.csv', es2_train, delimiter=',')
savetxt('es2_test.csv', es2_test, delimiter=',')
#scores = cross_val_score(model, X, y, scoring='accuracy', cv=5, n_jobs=-1)
#m, s = mean(scores), std(scores)

loaded_dataset = pd.read_csv('es2_test.csv', header=0)
changedVal=loaded_dataset.fillna(loaded_dataset.mode())
values = changedVal.values
X, y = values[:, :-1], values[:, -1]
X=X.astype('int64')
models, names = list(), list()


# cart
models.append(DecisionTreeClassifier())
names.append('CART')
# svm
models.append(SVC())
names.append('SVM')
# random forest
models.append(RandomForestClassifier())
names.append('RF')
# gbm
models.append(GradientBoostingClassifier())
names.append('GBM')
# evaluate models
#models.append(Sequential())
#names.append('Seq')
all_scores = list()
for i in range(len(models)):
	# create a pipeline for the model
	s = StandardScaler()
	p = Pipeline(steps=[('s',s), ('m',models[i])])
	scores = cross_val_score(p, X, y)
	all_scores.append(scores)
	# summarize
	m, s = mean(scores)*100, std(scores)*100
	print('%s %.3f%% +/-%.3f' % (names[i], m, s))
plt.boxplot(all_scores, labels=names)
plt.show()
X=np.array(X,dtype='float')
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.33)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test=X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()
model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]),go_backwards=True))
model.add(Dense(2, input_shape=(X_train.shape[1], X_train.shape[2])))
#model.add(LSTM(70,input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.1))
#model.add(LSTM(70))
#model.add(Dropout(0.3))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_absolute_error','accuracy'])
model.summary()
# fit network
history = model.fit(X_train, Y_train, epochs=400, batch_size=14, validation_data=(X_test, Y_test), verbose=1, shuffle=False)
#model.save('lstm_model.h5')


plt.plot(history.history['val_mean_absolute_error'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy wrt to loss')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['val_mean_absolute_error', 'loss'], loc='upper right')
plt.show()



plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy wrt to acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['accuracy', 'valACc'], loc='upper right')
plt.show()
# make a prediction
yhat = model.predict(X_train)

 
# load model
model = load_model('lstm_model.h5')
# summarize model.
model.summary()
# split into input (X) and output (Y) variables
# evaluate the model
score = model.evaluate(X_train, Y_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
print("%s: %.2f%%" % (model.metrics_names[2], score[2]*100))