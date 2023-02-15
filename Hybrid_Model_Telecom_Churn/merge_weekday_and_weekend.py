import graphlab as gl

w = gl.SFrame.read_csv("~/cdr/weekend.csv")

l1 = [2,3]
l2 = [9,10]
l3 = [16,17]
l4 = [23,24]

c = gl.SFrame.read_csv("~/cdr/week1.csv")
we = w.filter_by(l1,'day',exclude = False)
c = c.append(we)
c.save("week1.csv", format='csv')
print "completed week one"

c = gl.SFrame.read_csv("~/cdr/week2.csv")
we = w.filter_by(l2,'day', exclude = False)
c = c.append(we)
c.save("week2.csv",format='csv')
print "completed week two"

c = gl.SFrame.read_csv("~/cdr/week3.csv")
we = w.filter_by(l3,'day', exclude = False)
c = c.append(we)
c.save("week3.csv",format='csv')
print "completed week three"

c = gl.SFrame.read_csv("~/cdr/week1.csv")
we = w.filter_by(l4,'day', exclude = False)
c = c.append(we)
c.save("week4.csv",format='csv')
print "completed week four"

