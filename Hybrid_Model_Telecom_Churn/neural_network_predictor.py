import graphlab as gl
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow import metrics


cdr = gl.SFrame.read_csv("w1_features.csv")
print "Week 1 read"
mdl = gl.SFrame.read_csv("w2_features.csv")
cdr = cdr.append(mdl)
print "Week 2 read and appended"
mdl = gl.SFrame.read_csv("w3_features.csv")
cdr = cdr.append(mdl)
print "Week 3 read and appended"
mdl = gl.SFrame.read_csv("w4_features.csv")
cdr = cdr.append(mdl)
print "Week 4 read and appended"

cdr = cdr.to_dataframe()
print "SFrame converted to pandas dataframe"

X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16]].values
y = dataset.iloc[:, :10].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier=Sequential()

#first hidden layer
classifier.add(Dense(units=16,kernel_initializer=’uniform’,activation=’relu’,input_dim=16))

#second layer
classifier.add(Dense(units=11,kernel_initializer=’uniform’,activation=’relu’))

#output layer
classifier.add(Dense(units=1,kernel_initializer=’uniform’,activation=’sigmoid’))

#Compile the layers to form a model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['f_score'])

classifier.fit(X_train,y_train,batch_size=10,epochs=100)

pred = classifier.predict(X_test)

rcall = metrics.recall(y_test, pred)
prec = metrics.precision(y_test, pred)

f_score = (2*rcall*prec)/(rcall+prec)
print "The F-Score is ",f_score 
















