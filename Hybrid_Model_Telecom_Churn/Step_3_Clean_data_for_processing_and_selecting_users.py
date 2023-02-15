
# coding: utf-8

# In[ ]:

import graphlab as gl

#This code selects users who have made calls in all the weeks adn are less than 300, the range can be changed to observe that the results are consistent 

#Change file Name tp week1Final or week2Final to get the data for weekdays.

###cdr1=gl.SFrame.read_csv("/home/aiswarya/Sammy/allWeekends.csv")
cdr1=gl.SFrame.read_csv("weekdays1.csv")
incoming=cdr1.filter_by([0,2],'Call_Type')
outgoing=cdr1.filter_by([1,3],'Call_Type')
incoming.rename({'Callee': 'Customer'})
outgoing.rename({'Caller':'Customer'})
outgoing_calls_count=outgoing.groupby(key_columns=['Customer','Call_Type'],operations={'count': gl.aggregate.COUNT()})
incoming_calls_count=incoming.groupby(key_columns=['Customer','Call_Type'],operations={'count': gl.aggregate.COUNT()})
outgoing_calls_count.dropna('Customer')
len(outgoing_calls_count)
incoming_calls_count.dropna('Customer')
len(incoming_calls_count)
# here I kept the call count as 650 because this code counts for all weekends, March 2013 had 5 weekends so 650 accounts 
#for 10 days of data, For weekdays keep count limit to 280-300 for consistent results
outgoing_calls_count=outgoing_calls_count[outgoing_calls_count['count']<500]
calls=incoming_calls_count.join(outgoing_calls_count,on="Customer",how='inner')
calls.remove_columns({'Call_Type','count','count.1','Call_Type.1'})
calls.unique()
calls=calls['Customer']
incoming=incoming.filter_by(calls,'Customer')
outgoing=outgoing.filter_by(calls, 'Customer')
incoming.remove_column('Caller')
outgoing.remove_column('Callee')
cdr1=incoming.append(outgoing)
cdr1.save("weekdays1final_500.csv", format="csv")


# In[ ]:



