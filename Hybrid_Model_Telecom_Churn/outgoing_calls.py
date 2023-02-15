import graphlab as gl
import pandas as pd

c1 =  gl.SFrame.read_csv("~/cdr/week1.csv", usecols=['Customer','Duration','Callee'], column_type_hints=int)
c2 = gl.SFrame.read_csv("w1_features.csv")
c2 = c2.to_dataframe()

out_num =[]
out_dur =[]
sum = 0

x = ['Customer']
a = c2.select_columns(x)

for i in c2:
	d = c1.filter_by(i, 'Customer', exclude=False)
	out_num.append(len(d)
	for i in d['Duration']:
		sum = sum +i
	out_dur.append(sum)
	sum = 0

out_num = gl.SArray(data=out_num, dtype=int)
out_dur = gl.SArray(data=out_dur, dtype=int)

c2 = c2.add_column(out_num, 'No_of_outgoing_calls')
c2 = c2.add_column(out_dur, 'Outgoing_call_duration')

c2.save("w1_features.csv", format='csv')


