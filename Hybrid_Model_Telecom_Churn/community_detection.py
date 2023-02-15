import graphlab as gl
import pandas as pd
import networkx as nx
import community


#Community detection for Week 1
cdr = gl.SFrame.read_csv("~/cdr/w1_ed.csv", usecols=['Customer','Callee'],column_type_hints=int)
cdr = cdr.to_dataframe()
g = nx.Graph()
g = nx.from_pandas_dataframe(cdr, 'Customer','Callee',['Duration'])
f = open("w1_comm.csv","w")
comm = community.best_partition(g, resolution=0.6)
f.write(str(comm))
f.close()


#Community detection for Week 2
cdr = gl.SFrame.read_csv("~/cdr/w2_ed.csv", usecols=['Customer','Callee', 'Duration'],column_type_hints=int)
cdr = cdr.to_dataframe()
g = nx.Graph()
g = nx.from_pandas_dataframe(cdr, 'Customer','Callee',['Duration'])
f = open("w2_comm.csv","w")
comm = community.best_partition(g, resolution=0.6)
f.write(str(comm))
f.close()


#Community detection for Week 3
cdr = gl.SFrame.read_csv("~/cdr/w3_ed.csv", usecols=['Customer','Callee', 'Duration'],column_type_hints=int)
cdr = cdr.to_dataframe()
g = nx.Graph()
g = nx.from_pandas_dataframe(cdr, 'Customer','Callee',['Duration'])
f = open("w3_comm.csv","w")
comm = community.best_partition(g, resolution=0.6)
f.write(str(comm))
f.close()


#Community detection for Week 4
cdr = gl.SFrame.read_csv("~/cdr/w4_ed.csv", usecols=['Customer','Callee', 'Duration'],column_type_hints=int)
cdr = cdr.to_dataframe()
g = nx.Graph()
g = nx.from_pandas_dataframe(cdr, 'Customer','Callee',['Duration'])
f = open("w4_comm.csv","w")
comm = community.best_partition(g, resolution=0.6)
f.write(str(comm))
f.close()




