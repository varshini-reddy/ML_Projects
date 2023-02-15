import graphlab as gl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import f1_score

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
print "Week 4 read and appeneded"

cdr = cdr.to_dataframe()
print "SFrame converted to pandas dataframe"

features = cdr.drop(["Churn"], axis=1).columns
print "Feature selection"

cdr_train, cdr_test = train_test_split(cdr, test_size=0.25)
print "Data split to train and test sets"

mdl = tree.DecisionTreeClassifier()
mdl = mdl.fit(cdr_train[features], cdr_train["Churn"])
print "Decision tree model created"

prediction = mdl.predict(cdr_test[features])
#probability = mdl.predict_proba(cdr_test[features])

fs = f1_score(cdr_test["Churn"], prediction)
print "F-score of the model is", fs

 










