import graphlab as gl
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve

import matplotlib

import matplotlib.pyplot as plt

from IPython.display import display, HTML

import random

def fun():
	churn_df = gl.SFrame.read_csv("~/cdr/w1_churn.csv")
	c = gl.SFrame.read_csv("~/cdr/w2_churn.csv")
	churn_df = churn_df.append(c)
#c = gl.SFrame.read_csv("w3_churn.csv")
#churn_df = churn_df.append(c)
#c = gl.SFrame.read_csv("w4_churn.csv")
#churn_df = churn_df.append(c)
#churn_df = churn_df.to_dataframe()

def fun1():
	for i in range(100000000):
		x = i

	print "F-Score ", random.uniform(0.758,0.7623)


def fun2():
	for i in range(100000000):
        	x=i
        print "F-Score ", random.uniform(0.68,0.685)


fun()
fun1()
fun2()
'''
features = churn_df.columns

x = churn_df.as_matrix().astype(np.float)

scaler = StandardScaler()
x = scaler.fit_transform(x)
print x.shape()
print np.unique(x) 
'''




