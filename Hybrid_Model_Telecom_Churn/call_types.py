import graphlab as gl
import pandas as pd

c1 = gl.SFrame.read_csv("week1.csv", usecols=['Customer','Call_Type'], column_type_hints=int)
c2 = gl.SFrame.read_csv("w1_features.csv", column_type_hints=int)

Cust = c2.select_columns(['Customer'])
Cust = Cust.to_dataframe()
Cust = Cust.to_list()

ct1 = []
ct2 = []

for i in Cust:
	c = c1.filter_by(i, 'Customer', exclude=False)
	c = c.select_columns('Call_Type')
	c = c.to_list(c.to_dataframe())
	for j in c:	
		if j==0:
			c0 = c0 + 1
		else:
			c1 = c1 + 1

	ct1 = ct1.append(c0)
	ct2 = ct2.append(c1)

ct1 = gl.SArray(data=ct1, dtype=int)
ct2 = gl.SArray(data=ct2, dtype=int)

c2 = c2.add_columns('ct1','ct2', namelist=['No_of_type_0_calls','No_of_type_2_calls'])
c2.save("w1_features.csv", format='csv')
	


