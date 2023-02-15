import graphlab as gl

'''
c1 = gl.SFrame.read_csv("~/churn_prediction/data/Finalsubset_filter_cda.csv", usecols=["Customer"], column_type_hints=int)
c1 = list(c1["Customer"])

c2 = gl.SFrame.read_csv("~/churn_prediction/data/week1.csv", column_type_hints=int)
c2 = c2.append(gl.SFrame.read_csv("~/churn_prediction/data/week2.csv",column_type_hints=int))

cw = gl.SFrame.read_csv("~/churn_prediction/data/wend.csv",column_type_hints=int)

l = [2,3,9,10]
cw = cw.filter_by(l,'day',exclude=False)
c2 = c2.append(cw)
print len(c2)
c2 = c2.filter_by(c1,'Customer',exclude=False)
#print c1
print len(c2)

c2.save("Combined.csv",format='csv')
'''

cdr = gl.SFrame.read_csv("df16.csv", column_type_hints=int)
cdr.save("d16.csv", format='csv')

