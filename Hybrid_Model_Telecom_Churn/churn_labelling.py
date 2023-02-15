import graphlab as gl
import numpy as np
import csv

f = open("churners.csv","rU")
reader = csv.reader(f,delimiter = ",")

rownum = 0
a = np.empty([1],dtype=int)
print type(reader)

for row in reader:
	row = map(int, row)
        a = np.append(a,row)
        
cdr = gl.SFrame.read_csv("w1_features.csv")

churn = []
c1 = cdr.filter_by(a,'Customer',exclude=False)
for i in range(len(c1)):
	churn.append(1)
churn = gl.SArray(data=churn,dtype=int)
c1 = c1.add_column(churn, name='Churn')

churn = []
c2 = cdr.filter_by(a,'Customer',exclude=True)
for i in range(len(c2)):
	churn.append(0)
churn = gl.SArray(data=churn, dtype=int)
c2 = c2.add_column(churn, name='Churn')
c1 = c1.append(c2)
c1.save("w1_features.csv",format='csv')



