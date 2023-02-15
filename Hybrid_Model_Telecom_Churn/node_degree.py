import graphlab as gl
import pandas as pd

c = gl.SFrame.read_csv("week1.csv", usecols=['Customer','Callee'], column_type_hints=int)
c = c.to_dataframe()
g = nx.DiGraph()
g = nx.from_pandas_dataframe(c, 'Customer', 'Callee')

c = gl.SFrame.read_csv("w1_features.csv", column_type_hints=int)
c1 = c.select_columns(['Customer'])
c1 = c1.to_list(c1.to_dataframe))

in_deg = []
out_deg = []

for i in c1:
	in_deg.append(g.in_degree[i])
	out_degree.append(g.out_degree[i])

in_deg = gl.SArray(data=in_deg, dtype=int)
out_deg= gl.SArray(data=out_deg, dtype=int)
c = c.add_columns('in_deg', 'out_deg', namelist=['in_degree','out_degree'])
c.save("w1_features.csv", format='csv')




