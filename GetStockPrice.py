#import matplotlib.pyplot as plt
#import fix_yahoo_finance as yf  
#data = yf.download('AAPL','2016-01-01','2018-01-01')
#data.Close.plot()
#plt.show()

import pandas_datareader as web
import datetime as dt
import fix_yahoo_finance as yf

yf.pdr_override()

start = dt.datetime(2013, 1, 1)
end = dt.datetime(2017, 12, 31)

df = web.get_data_yahoo(['2317.TW'],start, end)
df.to_csv('2317.TW.csv')
