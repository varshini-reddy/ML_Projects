import networkx as nx
import graphlab as gl
import pandas as pd

c1 = gl.SFrame.read_csv("w1_comm.csv")
c1 = c1.to_dataframe()
#a = ['Customer']
#a = list(c1.select_columns(a))

c2 = gl.SFrame.read_csv("w1_ed.csv", usecols=['Customer','Callee'], column_type_hints=int)

c1 = c1.to_dict()
c1 = sorted(c1.items(), key=lambda x: x[1])

f = gl.SFrame.read_csv("w1_features.csv")
final = gl.SFrame()
centrality = []	

#Maximal social strength

for i in range(0,max(c1.values()):
	l = c1.keys()[c1.values().index(i)]
	c = c2.filter_by(l, 'Customer', exclude=False)				
	c = c.to_dataframe()
	g = nx.Graph()
	g = nx.from_pandas_dataframe(c)	
	cent = nx.eigenvector_centrality(g, max_iter=1000, tol=1, nstart=None)
	
	a = f.filter_by(cent.keys(), name='Customer', exclude=False)

	for j in a.select_columns(['Customer']):
		centrality = centrality.append(cent[i])
	
	final = final.append(a)	

centrality = gl.SArray(data=centrality,dtype=int)
final = final.add_column(centrality, name='max_social_strength')
final.save("w1_features.csv", format='csv') 



#This is for minimal social strength

final = gl.SFrame()
centrality = [] 

for i in range(0,max(c1.values())):
        l = c1.keys()[c1.values().index(i)]
        c = c2.filter_by(l, 'Customer', exclude=False)                          
        c = c.to_dataframe()
        g = nx.Graph()
        g = nx.from_pandas_dataframe(c) 
        cent = nx.eigenvector_centrality(g, max_iter=1000, tol=1e-02, nstart=None)
        
        a = f.filter_by(cent.keys(), name='Customer', exclude=False)
        
        for j in list(a.select_columns(['Customer'])):
                centrality = centrality.append(cent[i])
        
        final = final.append(a)

centrality = gl.SArray(data=centrality,dtype=int)
final = final.add_column(centrality, name='min_social_strength')
final.save("w1_features.csv", format='csv')



#Ratio of max and min social strength

cdr = gl.SFrame.read_csv("w1_features.csv", column_type_hints=int)
c = cdr.select_columns(['max_social_strength', 'min_social_strength'])
c = c.to_dataframe()
c = c.values.tolist()

ratio = []
for i in range(len(cdr)):
	ratio = ratio.append(c[i][0]/c[i][1])

ratio = gl.SArray(data=ratio, dtype=int)
cdr = cdr.add_column(ratio, name='ratio_of_social_strength')
cdr.save("w1_features.csv", format='csv')

	














