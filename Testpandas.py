import pandas as pd
import numpy as np
import datetime
tsharepdf=pd.read_csv('tsharepTest.csv',encoding = 'big5',thousands=',',usecols=["No","Date","Open","High","Low","Close","Volume"],low_memory=False)  #nrows=100000,,verbose=True

tsharepdf['Date'] = pd.to_datetime(tsharepdf['Date'],format='%Y%m%d') 
# columns = pd.MultiIndex.from_tuples([('1101', 'Open') ,('1101','High'),('1101',"Low"),('1101',"Close"),('1101',"Volume"), ('1102', 'Open') ,('1102','High'),('1102',"Low"),('1102',"Close"),('1102',"Volume")])
# tuples = list(zip(*[tsharepdf['No'],['one', 'two', 'one', 'two','one', 'two', 'one', 'two']]))
# 
# print(list(tsharepdf['No']))
# columns = pd.MultiIndex.from_product([['1101','1102'],["Open","High","Low","Close","Volume"]])
# print(columns)
# 
# df = pd.DataFrame(np.random.randn(8, 10), columns=columns)
# print(df)
# pivoted = pd.pivot_table(tsharepdf,columns=['No'],index=['Date'],values=["Open","High","Low","Close","Volume"])
# stacked=tsharepdf.stack('No')

pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_rows', 1000)

'''
change to what I want shape
'''
tsharepdf=pd.read_csv('tsharepTest.csv',encoding = 'big5',thousands=',',usecols=["No","Date","Open","High","Low","Close","Volume"],low_memory=False)  #nrows=100000,,verbose=True
tsharepdf['Date'] = pd.to_datetime(tsharepdf['Date'],format='%Y%m%d') 
print('=====Original=====')
print(tsharepdf)

print('=====pivoted=====')
pivoted=tsharepdf.pivot(index='Date', columns='No', values=["Close","High","Low","Open","Volume"])
# problem occur when stack level0 the sequence wrong
# pivoted=tsharepdf.pivot(index='Date', columns='No', values=["Open","High","Low","Close","Volume"])
print(pivoted)


print('=====forward fill na =====')
fill_pad_lim1=pivoted.fillna(method='pad',limit=3)
print(fill_pad_lim1)

print('=====fill zero in head=====')
fill_zero_in_head=fill_pad_lim1.fillna(0)
print(fill_zero_in_head)

# print('=====backward fill na =====')
# fill_pad_lim1=pivoted.fillna(method='bfill',limit=1)
# print(fill_pad_lim1)

print('=====stacked level0=====')
stacked=fill_zero_in_head.stack(level=0)
print(stacked)


print('=====unstacked=====')
unstacked=stacked.unstack()
print(unstacked)


print('=====stacked level1=====')
stacked=unstacked.stack(level=1)
print(stacked)

print('=====stacked ix 20130103=====')
print(stacked.ix['20130103'])

print('=====stacked loc=====')
print(stacked.loc['20130103'])

# print('=====select date range 20130102 to 20130104 =====')
# print(stacked.loc['20130102':'20130102'+pd.DateOffset(days=4)])


print('=====find week day of 20180608 =====')
date=datetime.date(2018,6,8)
print(date.isoweekday())

print('=====find date of 20180608+2days =====')
date=datetime.date(2018,6,8)
date=date+datetime.timedelta(days=2)
print(date)

print('=====stacked loc by datetime =====')
date=datetime.date(2013,1,3)
print(date)
print(stacked.loc[date])

print('=====date compare=====')
date1=datetime.date(2013,1,4)
date2=datetime.date(2013,1,6)
if date1 > date2:
    print('date1 > date2')
elif date1 < date2:
    print('date1 < date2')
else:
    print('date1 = date2')
    
print('=====is date in data=====')    
date1=datetime.date(2013,1,4)

if date1 in stacked.index:
    print("{0} is in data".format(date1))
else:
    print("{0} is not in data".format(date1))
    


print('=====unstacked loc 20130102 to 20130105=====')
date_start=datetime.date(2013,1,3)
date_end=date_start+datetime.timedelta(days=8)
print(date_start)
print(date_end)
print(unstacked.loc[date_start:date_end])

print('=====unstacked loc 20130102 to 20130105 to matrix=====')
date_start=datetime.date(2013,1,3)
date_end=date_start+datetime.timedelta(days=1)
print(date_start)
print(date_end)
print(unstacked.loc[date_start:date_end].as_matrix())
print(unstacked.loc[date_start:date_end].as_matrix().shape)

print('=====pivoted just close =====')
pivoted_jc=tsharepdf.pivot(index='Date', columns='No', values=["Close"])
print(pivoted_jc)

print('=====forward fill na just close =====')
fill_pad_lim1_jc=pivoted_jc.fillna(method='pad',limit=3)
print(fill_pad_lim1_jc)

print('=====fill zero in head just close=====')
fill_zero_in_head_jc=fill_pad_lim1_jc.fillna(0)
print(fill_zero_in_head_jc)

print('=====stack just close=====')
stacked_jc=fill_zero_in_head_jc.stack(level=1)
print(stacked_jc)
print(stacked_jc.as_matrix().shape)


print('=====stacked just close loc 20130102 to 20130105=====')
date_start=datetime.date(2013,1,3)
date_end=date_start+datetime.timedelta(days=2) #if over the level mismatch error occur
print(date_start)
print(date_end)
print(stacked_jc.loc[date_start:date_end])


# print('=====stacked just close drop raw=====')
# stacked_jc=stacked_jc.drop(pd.to_datetime('20130104'))
# print(stacked_jc)
# print(pivoted)
# print(tsharepdf)
# 
# multiIndex = pd.MultiIndex.from_arrays([['1101','1101','1101','1101','1101','1102','1102','1102','1102','1102'],["Open","High","Low","Close","Volume","Open","High","Low","Close","Volume"]])
# print(multiIndex)

# df1=tsharepdf.groupby(['No'])
# print(df1.groups)
# print(df1.get_group(1101))
# 
# df2=tsharepdf.groupby(['No'])
# print(df2.get_group(1102))
# 
# # result = pd.concat([df1.get_group(1101), df1.get_group(1102)], axis=1)
# print(result)


# tsharepdf.set_index(keys=['Date','No'],inplace=True)
# print(tsharepdf['20130102':'20130103'])

# s = pd.Series([1,3,5,np.nan,6,8])
# print(s)
# 
# dates=pd.date_range('20130101',periods=6)
# print(dates)
# 
# df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
# print(df)
# 
# pieces=[df[:3],df[3:7],df[7:]]
# print(pd.concat(pieces))
# s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(1)
# print(s)



# df2 = pd.DataFrame({ 
# 'A' : 1.,
# 'B' : pd.Timestamp('20130102'),
# 'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
# 'D' : np.array([3] * 4,dtype='int32'),
# 'E' : pd.Categorical(["test","train","test","train"]),
# 'F' : 'foo' })
# 
# print(df2.index)
# 
# df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
# 'foo', 'bar', 'foo', 'foo'],
# 'B' : ['one', 'one', 'two', 'three',
# 'two', 'two', 'one', 'three'],
# 'C' : np.random.randn(8),
# 'D' : np.random.randn(8)})
# 
# print(df)
# 
# print(df.groupby(['A','B']))


# for i,k in df2.items():
#     print(i)
#     print(k)
# print(df2.items())

#print(tsharepdf['20180403':'20180504'])


#     print(tsharepdf.shape)
#     
#     print(tsharepdf['20180418'].head(10))
#     
#     print('=======date range========')
# print(tsharepdf['20180403':'20180504'])
# print(tsharepdf.ix['20180403'])
#     
# df0403to0504=tsharepdf['20180403':'20180504']
#     print(df0403to0504['No']==1101)
    
# filter=(tsharepdf['No']==1101)
# print(tsharepdf[filter])
# 
# mask=(tsharepdf['No']<=1101)
# print(tsharepdf[mask])
# 
# mask1=(tsharepdf['No']<=1104)
# mask2=((tsharepdf['Date']>=20130103) & (tsharepdf['Date']<= 20130109))
# print(tsharepdf[(mask1 & mask2)])
# 
# print(tsharepdf[tsharepdf['Date'].between(20130102,20130103)])
# print(tsharepdf.info)
# 
# # print(tsharepdf['Date'].[0:4])
# print(tsharepdf[['Date','No']])
# 
# print(tsharepdf.sort_values("Date"))
