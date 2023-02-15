import graphlab as gl
import numpy
import pdb


def unique_of_w1_w2():	
	cdr1 = gl.SFrame.read_csv("week1.csv")
	l = ['Customer']
	cdr1 = cdr1.select_columns(l)
	c1 = cdr1.to_numpy()
	cdr2 = gl.SFrame.read_csv("week2.csv")
	cdr2 = cdr2.select_columns(l)
	c2 = cdr2.to_numpy()

	c = numpy.union1d(c1,c2)
	return c


def unique_of_w3_w4():
        cdr1 = gl.SFrame.read_csv("week3.csv")
        l = ['Customer']
        cdr1 = cdr1.select_columns(l)
        c1 = cdr1.to_numpy()
        cdr2 = gl.SFrame.read_csv("week4.csv")
        cdr2 = cdr2.select_columns(l)
        c2 = cdr2.to_numpy()

        c = numpy.union1d(c1,c2)
        return c

	
def churner_list():
	a = unique_of_w1_w2()
	b = unique_of_w3_w4()
	c = numpy.setdiff1d(a,b)
	#numpy.savetxt("churn.csv",c,delimiter=",")
	c.tofile("churners.csv",sep=",",format="%d")


churner_list()









