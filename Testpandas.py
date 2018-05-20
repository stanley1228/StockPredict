import pandas as pd


tsharepdf=pd.read_csv('tsharepEnTitle.csv',encoding = 'big5',thousands=',',usecols=["No","Date","Open","High","Low","Close","Volume"],low_memory=False)  #nrows=100000,,verbose=True

tsharepdf['Date'] = pd.to_datetime(tsharepdf['Date'],format='%Y%m%d') 
tsharepdf = tsharepdf.set_index('Date',inplace=true)
#     print(tsharepdf.head(5))
#     print(tsharepdf.tail(5))
#     print(tsharepdf.shape)
#     
#     print(tsharepdf['20180418'].head(10))
#     
#     print('=======date range========')
#     print(tsharepdf['20180403':'20180504'])
#     print(tsharepdf.ix['20180403'])
    
df0403to0504=tsharepdf['20180403':'20180504']
#     print(df0403to0504['No']==1101)
    
filter=(tsharepdf['No']==1101)
print(tsharepdf[filter])

mask=(tsharepdf['No']<=1101)
print(tsharepdf[mask])

mask1=(tsharepdf['No']<=1104)
mask2=((tsharepdf['Date']>=20130103) & (tsharepdf['Date']<= 20130109))
print(tsharepdf[(mask1 & mask2]))

print(tsharepdf[tsharepdf['Date'].between(20130102,20130103)])
print(tsharepdf.info)

print(tsharepdf['Date'].[0:4])
print(tsharepdf[['Date','No']])

print(tsharepdf.sort_values("Date"))
