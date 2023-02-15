import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import graphlab as gl

df = gl.SFrame.read_csv("~/programs/w1_features.csv")
df = pd.DataFrame(df)
print "Converted SFrame to pandas dataframe"

features = df.drop(["Churn"], axis=1).columns
print "The features are ", features

df_train, df_test = train_test_split(df, test_size=0.25)
print "Data split to train and test"

clf = RandomForestClassifier(n_estimators=14)
print "Model constructed"

clf = clf.fit(df_train[features], df_train["Churn"])
print "Model fitting"

predictions = clf.predict(df_test[features])
print "Model prediction is ", predictions

#probs = clf.predict_proba(df_test[features])
#print "Model prediction probability is ", probs

score = f1_score(df_test["Churn"], predictions)
print "F-Score: ", score



