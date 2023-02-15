
# coding: utf-8

# In[ ]:

import graphlab as gl
cdr1=gl.SFrame.read_csv("../data/01_new.csv")


##THe CSV file doesnt have column names so this code adds another row to the csv to give it column names
sf=gl.SFrame({'1': [1] ,'10792228': [10792228] ,'18154720': [18154720] ,'2013-03-01 21:04:55' : ['2013-03-01 21:04:55'],'124': [124] ,'2731712': [2731712] ,'508306': [508306] ,'101': [101] , '101.1': [101], '0' : [0]})

cdr1.append(sf)
cdr1.rename({'1': 'Call_Type' ,'10792228': 'Caller' ,'18154720': 'Callee' ,'2013-03-01 21:04:55' : 'timestamp','124': 'Duration' ,'2731712': 'X6' ,'508306': 'X7' ,'101': 'TowerID_start' , '101.1': 'TowerID_end', '0' : 'Call_Status'})

cdr1.remove_columns(['X6','X7'])

cdr1.remove_column('Call_Status')

##to split date-time into days hours minutes and seconds
cdr1['date-time']=cdr1['timestamp'].str_to_datetime('%Y-%m-%d %H:%M:%S')
cdr1=cdr1.split_datetime('date-time',limit=['day','hour','minute','second'])
#cdr1.remove_column('X4')

cdr1.rename({'date-time.hour':'call hour','date-time.minute':'call minute','date-time.second':'call seconds','date-time.day':'day'})


cdr1.save("cdr1_churn.csv", format="csv")


#cdr2_6=gl.SFrame.read_csv("../data/02_06_new.csv")
#
#
#sf=gl.SFrame({'2': [2] ,'9555': [9555] ,'22470999': [22470999] ,'2013-03-02 17:51:21' : ['2013-03-02 17:51:21'],'1': [1] ,'2332273': [2332273] ,'2452790': [2452790] ,'4428': [4428] , '4428.1': [4428], '0' : [0]})
#
#cdr2_6.append(sf)
#
#cdr2_6.rename({'2': 'Call_Type' ,'9555': 'Caller' ,'22470999': 'Callee' ,'2013-03-02 17:51:21' : 'X4','1': 'Duration' ,'2332273': 'X6' ,'2452790': 'X7' ,'4428': 'TowerID_start' , '4428.1': 'TowerID_end', '0' : 'Call_Status'})
#
#cdr2_6['date-time']=cdr2_6['X4'].str_to_datetime('%Y-%m-%d %H:%M:%S')
#cdr2_6=cdr2_6.split_datetime('date-time',limit=['day','hour','minute','second'])
##cdr1.remove_column('X4')
#
#
#
##cdr2_6['date-time.day'].sketch_summary()
#
#cdr2_6.remove_columns(['X4','X6','X7','Call_Status'])
#
#
#
#cdr2_6.rename({'date-time.hour':'call hour','date-time.minute':'call minute','date-time.second':'call seconds'})
#
#cdr2_6.rename({'date-time.day':'day'})
#
#cdr2_6.save("cdr6.csv", format="csv")



#cdr7_10=gl.SFrame.read_csv("../data/07_10_new.csv")
#cdr7_10
#
#sf=gl.SFrame({'1': [1] ,'36461283': [36461283] ,'36461286': [36461286] ,'2013-03-07 20:38:11' : ['2013-03-07 20:38:11'],'23': [23] ,'1670207': [1670207] ,'3967105': [3967105] ,'6338': [6338] , '6338.1': [6338], '0' : [0]})
#
#cdr7_10.append(sf)
#
#cdr7_10.rename({'1': 'Call_Type' ,'36461283': 'Caller' ,'36461286': 'Callee' ,'2013-03-07 20:38:11' : 'X4','23': 'Duration' ,'1670207': 'X6' ,'3967105': 'X7' ,'6338': 'TowerID_start' , '6338.1': 'TowerID_end', '0' : 'Call_Status'})
#
#cdr7_10['date-time']=cdr7_10['X4'].str_to_datetime('%Y-%m-%d %H:%M:%S')
#cdr7_10=cdr7_10.split_datetime('date-time',limit=['day','hour','minute','second'])
#
#cdr7_10.remove_columns(['X4','X6','X7','Call_Status'])
#
#cdr7_10.rename({'date-time.hour':'call hour','date-time.minute':'call minute','date-time.second':'call seconds','date-time.day':'day'})
#
#cdr7_10.save("cdr10.csv", format="csv")
#
#
#

#cdr16_20=gl.SFrame.read_csv("../data/16_20_new.csv")
#
#cdr16_20
#
#sf=gl.SFrame({'3': [3] ,'27349717': [27349717] ,'26392517': [26392517] ,'2013-03-16 00:40:52' : ['2013-03-16 00:40:52'],'1': [1] ,'1595413': [1595413] ,'3163499': [3163499] ,'4340': [4340] , '4340.1': [4340], '1.1' : [1]})
#
#cdr16_20.append(sf)
#
#cdr16_20.rename({'3': 'Call_Type' ,'27349717': 'Caller' ,'26392517': 'Callee' ,'2013-03-16 00:40:52' : 'X4','1': 'Duration' ,'1595413': 'X6' ,'3163499': 'X7' ,'4340': 'TowerID_start' , '4340.1': 'TowerID_end', '1.1' : 'Call_Status'})
#
#
#
#cdr16_20['date-time']=cdr16_20['X4'].str_to_datetime('%Y-%m-%d %H:%M:%S')
#cdr16_20=cdr16_20.split_datetime('date-time',limit=['day','hour','minute','second'])
#
#cdr16_20.remove_columns(['X4','X6','X7','Call_Status'])
#
#cdr16_20.rename({'date-time.hour':'call hour','date-time.minute':'call minute','date-time.second':'call seconds','date-time.day':'day'})
#
#
#cdr16_20.save("cdr20.csv", format="csv")
#
#
#
#
#cdr21_23=gl.SFrame.read_csv("../data/21_23_new.csv")
#
#sf=gl.SFrame({'2': [2] ,'27399645': [27399645] ,'3587670': [3587670] ,'2013-03-21 00:06:44' : ['2013-03-21 00:06:44'],'1': [1] ,'1807916': [1807916] ,'2769836': [2769836] ,'5942': [5942] , '5942.1': [5942], '1.1' : [1]})
#
#cdr21_23.append(sf)
#
#cdr21_23.rename({'2': 'Call_Type' ,'27399645': 'Caller' ,'3587670': 'Callee' ,'2013-03-21 00:06:44' : 'X4','1': 'Duration' ,'1807916': 'X6' ,'2769836': 'X7' ,'5942': 'TowerID_start' , '5942.1': 'TowerID_end', '1.1' : 'Call_Status'})
#
#cdr21_23['date-time']=cdr21_23['X4'].str_to_datetime('%Y-%m-%d %H:%M:%S')
#cdr21_23=cdr21_23.split_datetime('date-time',limit=['day','hour','minute','second'])
#
#cdr21_23.remove_columns(['X4','X6','X7','Call_Status'])
#
#cdr21_23.rename({'date-time.hour':'call hour','date-time.minute':'call minute','date-time.second':'call seconds','date-time.day':'day'})
#
#
#cdr21_23.save("cdr23.csv", format="csv")
#
#
#cdr24_27=gl.SFrame.read_csv("../data/24_27_new.csv")
#
#
#
#sf=gl.SFrame({'0': [0] ,'35218309': [35218309] ,'36242131': [36242131] ,'2013-03-24 10:19:19' : ['2013-03-24 10:19:19'],'60': [60] ,'1743574': [1743574] ,'3530386': [3530386] ,'7452': [7452] , '7452.1': [7452], '1' : [1]})
#
#cdr24_27.append(sf)
#
#cdr24_27.rename({'0': 'Call_Type' ,'35218309': 'Caller' ,'36242131': 'Callee' ,'2013-03-24 10:19:19' : 'X4','60': 'Duration' ,'1743574': 'X6' ,'3530386': 'X7' ,'7452': 'TowerID_start' , '7452.1': 'TowerID_end', '1' : 'Call_Status'})
#
#cdr24_27['date-time']=cdr24_27['X4'].str_to_datetime('%Y-%m-%d %H:%M:%S')
#cdr24_27=cdr24_27.split_datetime('date-time',limit=['day','hour','minute','second'])
#
#cdr24_27.remove_columns(['X4','X6','X7','Call_Status'])
#
#cdr24_27.rename({'date-time.hour':'call hour','date-time.minute':'call minute','date-time.second':'call seconds','date-time.day':'day'})
#
#cdr24_27.save("cdr27.csv", format="csv")
#
#
#cdr28_31=gl.SFrame.read_csv("../data/28_31_new.csv")


#sf=gl.SFrame({'3': [3] ,'18677080': [18677080] ,'22694366': [22694366] ,'2013-03-31 23:42:00' : ['2013-03-31 23:42:00'],'1': [1] ,'1193584': [1193584] ,'18287': [18287] ,'3809': [3809] , '3809.1': [3809], '1.1' : [1]})

#cdr28_31.append(sf)

#cdr28_31.rename({'3': 'Call_Type' ,'18677080': 'Caller' ,'22694366': 'Callee' ,'2013-03-31 23:42:00' : 'X4','1': 'Duration' ,'1193584': 'X6' ,'18287': 'X7' ,'3809': 'TowerID_start' , '3809.1': 'TowerID_end', '1.1' : 'Call_Status'})

#cdr28_31['date-time']=cdr28_31['X4'].str_to_datetime('%Y-%m-%d %H:%M:%S')
cdr28_31=cdr28_31.split_datetime('date-time',limit=['day','hour','minute','second'])

#cdr28_31.remove_columns(['X4','X6','X7','Call_Status'])

#cdr28_31.rename({'date-time.hour':'call hour','date-time.minute':'call minute','date-time.second':'call seconds','date-time.day':'day'})

#cdr28_31.save("cdr31.csv", format="csv")







