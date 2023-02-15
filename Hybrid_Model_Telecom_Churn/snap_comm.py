import snap
import networkx as nx
import graphlab as gl
import pdb
'''
cdr = gl.SFrame.read_csv("Combined.csv", usecols=['Customer','Callee'],column_type_hints=int)
cdr = cdr.to_dataframe()
g = nx.Graph()
g = nx.from_pandas_dataframe(cdr, 'Customer','Callee',['Duration'])
f = open("w1_comm.csv","w")
comm = community.best_partition(g, resolution=0.6)
f.write(str(comm))
f.close()


'''

pdb.set_trace()
g = snap.LoadEdgeList(snap.PUNGraph, "combined.txt",1,2)
Cmty = snap.TCnComV()
mod = snap.CommunityGirvanNewman(g, Cmty)
for i in Cmty:
	for j in i:
		print j

print "Modularity is ", mod



