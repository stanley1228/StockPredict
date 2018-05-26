import pandas as pd
import numpy as np

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


'''
change to what I want shape
'''
tsharepdf=pd.read_csv('tsharepTest.csv',encoding = 'big5',thousands=',',usecols=["No","Date","Open","High","Low","Close","Volume"],low_memory=False)  #nrows=100000,,verbose=True
tsharepdf['Date'] = pd.to_datetime(tsharepdf['Date'],format='%Y%m%d') 

pivoted=tsharepdf.pivot(index='Date', columns='No', values=["Open","High","Low","Close","Volume"])
print(pivoted)

stacked=pivoted.stack(level=0)
print(stacked)

unstacked=stacked.unstack()
print(unstacked)


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
