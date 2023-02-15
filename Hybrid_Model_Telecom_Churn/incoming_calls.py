import graphlab as gl
import pandas as pd

c1 =  gl.SFrame.read_csv("~/cdr/week1.csv", usecols=['Customer','Duration','Callee'], column_type_hints=int)
c2 = gl.SFrame.read_csv("w1_features.csv")
c2 = c2.to_dataframe()

inc_num =[]
inc_dur =[]
sum = 0

x = ['Customer']
a = c2.select_columns(x)

for i in c2:
        d = c1.filter_by(i, 'Callee', exclude=False)
        inc_num.append(len(d)
        for i in d['Duration']:
                sum = sum +i
        inc_dur.append(sum)
        sum = 0

inc_num = gl.SArray(data=inc_num, dtype=int)
inc_dur = gl.SArray(data=inc_dur, dtype=int)

c2 = c2.add_column(inc_num, 'No_of_incoming_calls')
c2 = c2.add_column(inc_dur, 'Incoming_call_duration')

c2.save("w1_features.csv", format='csv')
