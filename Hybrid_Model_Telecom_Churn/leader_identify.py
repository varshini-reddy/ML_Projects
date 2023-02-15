import graphlab as gl
import pandas as pd

f = open("leader.txt", "w+")
c1 = gl.SFrame.read_csv("w1_comm.csv")
c1 = c1.to_dataframe()

c2 = gl.SFrame.read_csv("~/cdr/w1_ed.csv", usecols=['Customer','Callee'])

l1 = []
l2 = []
count = 1

c1 = sorted(c1.items(), key=lambda x: x[1])

for node,comm in c1.items():
	if count == int(comm):
		l1.append(node)
	else:
		l2.append(l)
		count = count + 1
		l1 = []

l1 = {}
leader = []
for i in l2:
	a = c2.filter_by(i, column_name='Customer',exclude=False)
	a = a.to_dataframe()
	g = nx.Graph()
	g = nx.from_pandas_dataframe(a, 'Customer', 'Callee')
	for j in i:
		deg = g.degree([j])
		l1[deg] = j
	leader = leader.append(l1[max(l1.keys()])
	l1 = {}

f.write(leader)

c2 = gl.SFrame.read_csv("w1_feature.csv")

l2 = []
c = c2.filter_by(leader, 'Customer', exclude=False)
for i in c:
	l2.append(1)

l2 = gl.SArray(data=l2, dtype=int)
c = c.add_column(l2, name='leader')

l2 = [] 
c2 = c2.filter_by(leader,'Customer', exclude=True)
for i in c2:
	l2.append(0)

l2 = gl.SArray(data=l2, dtype=int)
c2 = c2.add_column(l2, name='leader')

c = c.append(c2)
c.save("w1_feature.csv", format='csv')
		

	
	


