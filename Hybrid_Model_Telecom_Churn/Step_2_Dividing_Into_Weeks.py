
# coding: utf-8

# In[2]:

import graphlab as gl
cdr6=gl.SFrame.read_csv("/home/aiswarya/Sammy/cdr10.csv")

cdr1=gl.SFrame.read_csv("/home/aiswarya/Sammy/cdr1.csv")


cdr20=gl.SFrame.read_csv("/home/aiswarya/Sammy/cdr20.csv")

cdr31=gl.SFrame.read_csv("/home/aiswarya/Sammy/cdr31.csv")

cdr10=gl.SFrame.read_csv("/home/aiswarya/Sammy/cdr6.csv")

cdr1.add_column('day')

cdr10=cdr10.append(cdr1)

cdr10=cdr10.append(cdr6)

cdr10['day'].sketch_summary()

#Since 1 fell on a friday I took it in the first week itself,  the following code filters the csv files generated in the previous
#step and divides them into weeks and weekends

days=[1,4,5,6,7,8]
week1cdr=cdr10.filter_by(days,'day')

weekendcdr=cdr10.filter_by(days,'day',exclude=True)


cdr10=cdr20

cdr23=gl.SFrame.read_csv("/home/aiswarya/Sammy/cdr23.csv")

cdr10=cdr10.append(cdr23)

days=[18,19,20,21,22]
week2cdr=cdr10.filter_by(days,'day')

weekendtempcdr=cdr10.filter_by(days,'day',exclude=True)


weekendcdr=weekendcdr.append(weekendtempcdr)

days=[24,30,31]
week4cdr=cdr31.filter_by(days,'day',exclude=True)


weekendtemp=cdr31.filter_by(days,'day')

weekendcdr=weekendcdr.append(weekendtemp)



weekendcdr.save("/home/aiswarya/Sammy/allWeekends.csv",format="csv")

week1cdr.save("/home/aiswarya/Sammy/week1.csv", format="csv")
week2cdr.save("/home/aiswarya/Sammy/week2.csv", format="csv")
week4cdr.save("/home/aiswarya/Sammy/week4.csv", format="csv")

