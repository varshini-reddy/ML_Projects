import graphlab as gl
import pandas as pd


c1 = gl.SFrame.read_csv("w1_comm.csv")
c1 = c1.to_dataframe()

c2 = gl.SFrame.read_csv("w1_ed.csv", usecols=['Customer','Callee'], column_type_hints=int)

c1 = c1.to_dict()
c1 = sorted(c1.items(), key=lambda x: x[1])

f = gl.SFrame.read_csv("w1_features.csv")
final = gl.SFrame()
centrality = []

for i in range(0,max(c1.values()):
        l = c1.keys()[c1.values().index(i)]
        c = c2.filter_by(l, 'Customer', exclude=False)
        c = c.to_dataframe()
        g = nx.Graph()
        g = nx.from_pandas_dataframe(c)
        cent = nx.betweenness_centrality(g, normalized=True)

        a = f.filter_by(cent.keys(), name='Customer', exclude=False)

        for j in a.select_columns(['Customer']):
                centrality = centrality.append(cent[i])

        final = final.append(a)

centrality = gl.SArray(data=centrality,dtype=int)
final = final.add_column(centrality, name='betweenness_centrality')
final.save("w1_features.csv", format='csv')







