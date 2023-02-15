import graphlab as gl
import pandas as pd

c = gl.SFrame.read_csv("w1_comm.csv")
c = c.to_dict(c.to_dataframe())
c = sorted(c.items(), key=lambda x: x[1])


m = max(c.values())
l1 = []

for i in m:
	l1 = l1.append(list(c.keys()[c.values().index(i)]))

c = gl.SFrame.read_csv("week1.csv")
cdr = gl.SFrame.read_csv("w1_features.csv")
c2 = c2.to_list(c2.to_dataframe(cdr.select_columns('Customer')))
count = 0
co = []
final = []

for i in l1:
	x = c.filter_by(i, name='Customer', exclude=False)
	g = nx.Graph()
	g = nx.from_pandas_dataframe(x.to_dataframe())
	coeff = nx.jaccard_coefficient(G, ebunch=None)
	
	for j in coeff: 
		if j[0]==c2[count]:
			while(j[0]==c2[count]):
				co = co.append(j[2])	
			final = final.append(mean(co))
			count = count+1
	
final = gl.SArray(data=final, dtype=float)
cdr = cdr.add_column(final, name='Similarity')
cdr.save("w1_features.csv", format='csv')
	
	
	
	
	
	
	
	
