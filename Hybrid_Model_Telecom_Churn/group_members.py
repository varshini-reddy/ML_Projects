import graphlab as gl
import pandas as pd
from collections import defaultdict

cdr = gl.SFrame.read_csv("w1_comm.csv")
cdr = cdr.to_dataframe()

l1 = []

cdr = cdr.to_dict()
cdr = sorted(cdr.items(), key=lambda x: x[1])

m = max(cdr.values())

for i in range(m):
	l1 = l1.append(list(cdr.keys()[cdr.values().index(i)]))


cdr = gl.SFrame.read_csv("w1_ed.csv", usecols=['Customer'], column_type_hints=int)
cdr = cdr.to_dataframe()
cdr = cdr.unique()
comm_mem = []


count = 0
for cust in cdr:
	for i in l1:
		for j in i:
			if j==cust:
				comm_mem.append(len(i)-1)

l1 = gl.SArray(data=comm_mem, dtype=int)			
cdr = gl.SFrame(data=cdr)
cdr = cdr.add_column(l1, name='No_of_members')
cdr.save("w1_feature.csv", format='csv')		



