import pandas as pd

import numpy as np

from sklearn import preprocessing

def normalize(df):
    newdf=df.copy()
    min_max_scaler = preprocessing.MinMaxScaler()
    
    newdf['Open']=min_max_scaler.fit_transform(df['Open'].values.reshape(-1,1))
    newdf['High']=min_max_scaler.fit_transform(df['High'].values.reshape(-1,1))
    newdf['Low']=min_max_scaler.fit_transform(df['Low'].values.reshape(-1,1))
    newdf['Close']=min_max_scaler.fit_transform(df['Close'].values.reshape(-1,1))
    #newdf['Adj Close']=min_max_scaler.fit_transform(df['Adj Close'].values.reshape(-1,1))
    newdf['Volume']=min_max_scaler.fit_transform(df['Volume'].values.reshape(-1,1))
    
    return newdf

def data_helper(df,time_frame):
    number_features=len(df.columns)
    
    datavalue= df.as_matrix()
    
    result=[]
    
    for index in range(len(datavalue)-(time_frame+1)):
        result.append(datavalue[index: index+(time_frame+1)])
    
    result = np.array(result)
    number_train=round(0.9*result.shape[0])
    
    #train data
    x_train=result[:int(number_train),:-1]
    y_train=result[:int(number_train),-1][:,-1]
    
    print(x_train.shape)
    print(y_train.shape)
    
    #test data
    x_test=result[int(number_train):,:-1]
    y_test=result[int(number_train):,-1][:,-1]
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], number_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], number_features))  
    
    return [x_train, y_train, x_test, y_test]

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.recurrent  import LSTM
import keras

def build_model(input_length,input_dim):
    d=0.3
    model=Sequential()
    
    model.add(LSTM(256,input_shape=(input_length,input_dim),return_sequences=True))
    model.add(Dropout(d))
    
    model.add(LSTM(256,input_shape=(input_length,input_dim),return_sequences=False))
    model.add(Dropout(d))

    model.add(Dense(16,kernel_initializer="uniform",activation='relu'))
    model.add(Dense(1,kernel_initializer="uniform",activation='linear'))
    
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    
    return model

def denormalize(df,norm_value):
    original_value=df['Close'].values.reshape(-1,1)
    norm_value=norm_value.reshape(-1,1)
    
    min_max_scaler=preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm_value=min_max_scaler.inverse_transform(norm_value)
    
    return denorm_value



import matplotlib.pyplot as plt 
foxconndf=pd.read_csv('./2317_TW.csv',index_col=0)
foxconndf.dropna(how='any',inplace=True)
#close=foxconndf['Close']
#plt.plot(close)
#plt.show()
foxconndf.head();

foxconndf_norm=normalize(foxconndf)
model=build_model(20,5)

x_train, y_train, x_test, y_test = data_helper(foxconndf_norm, 20)

model.fit(x_train, y_train, batch_size=128,epochs=50,validation_split=0.1,verbose=1)

pred_train=model.predict(x_train)
denorm_pred_train=denormalize(foxconndf,pred_train)
denorm_ytest_train=denormalize(foxconndf,y_train)

pred=model.predict(x_test)
denorm_pred=denormalize(foxconndf,pred)
denorm_ytest=denormalize(foxconndf,y_test)


plt.figure(1)
plt.plot(denorm_pred_train,color='red',label='Prediction')
plt.plot(denorm_ytest_train,color='blue',label='Answer')
plt.legend(loc='best')

plt.figure(2)
plt.plot(denorm_pred,color='red',label='Prediction')
plt.plot(denorm_ytest,color='blue',label='Answer')
plt.legend(loc='best')
plt.show()

#print(tsmcdf_norm)
#close=tsmcdf['Close']
#close.plot()
#plt.plot(close)
#plt.show()