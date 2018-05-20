import pandas as pd

import numpy as np

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.recurrent  import LSTM
from keras.models import Model
# import keras
import matplotlib.pyplot as plt 

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

def data_helper(tsharepdf,tetfpdf,day_frame):
    
    number_features=5
    pred_days=5 #predict future 5 days

    '''share'''
    
    #find how many days of the data
    StocksGroup=tsharepdf.groupby('No')
    Group1101=StocksGroup.get_group(1101)
    data_days=len(Group1101)

    x_train_list=[]
    x_test_list=[]
    for stock_no,OneStockdf in StocksGroup:
#         if stock_no==1102:
#             break
        if len(OneStockdf) != data_days:
            continue
        OneStockFrameData=[]
        OneStock_mx=OneStockdf.as_matrix()
        for index in range(data_days-(day_frame+pred_days)+1):
            OneStockFrameData.append(OneStock_mx[index:index+day_frame,2:8]) #"No","Date","Name","Open","High","Low","Close","Volume"     
        OneStockFrameData=np.array(OneStockFrameData)
        OneStockFrameData=np.reshape(OneStockFrameData,(OneStockFrameData.shape[0],OneStockFrameData.shape[1],number_features))
        
        number_train=round(0.5*OneStockFrameData.shape[0])

        #x train data
        x_train_list.append(OneStockFrameData[:int(number_train)])   
        
        #x test data
        x_test_list.append(OneStockFrameData[int(number_train):])
    
    x_train_np_matrix=np.concatenate(x_train_list,axis=2)
    print(x_train_np_matrix.shape)
    x_train_np_matrix.tofile('x_train_np_matrix.dat')
    
    x_test_np_matrix=np.concatenate(x_test_list,axis=2) 
    print(x_test_np_matrix.shape)
    x_test_np_matrix.tofile('x_test_np_matrix.dat')
    
    
    '''ETF''' 
    ETFGroup=tetfpdf.groupby('No')
    
    y_train_list=[]
    y_test_list=[]

    for ETF_no,OneETFdf in ETFGroup:
        OneETFFrameData=[]
        OneETF_mx=OneETFdf.as_matrix()  
#         if ETF_no==690:
#             ETF_no=ETF_no
            
        if len(OneETFdf) < data_days:
            zero_m=np.zeros((data_days-len(OneETFdf),OneETF_mx.shape[1]))
            OneETF_mx=np.insert(arr=OneETF_mx, obj=0, values=zero_m, axis=0)
        for index in range(day_frame,OneETF_mx.shape[0]-pred_days+1):
            OneETFFrameData.append(OneETF_mx[index:index+pred_days,5]) #"No","Date","Name","Open","High","Low","Close","Volume"     
        OneETFFrameData=np.array(OneETFFrameData)
     
        OneETFFrameData=np.reshape(OneETFFrameData,(OneETFFrameData.shape[0],OneETFFrameData.shape[1],1))
        
        #y train data
        y_train_list.append(OneETFFrameData[:int(number_train)])                           
      
        #y test data
        y_test_list.append(OneETFFrameData[int(number_train):])

    y_train_np_matrix=np.concatenate(y_train_list,axis=1)
    print(y_train_np_matrix.shape)
    y_train_np_matrix.tofile('y_train_np_matrix.dat')
   
    y_test_np_matrix=np.concatenate(y_test_list,axis=1)
    print(y_test_np_matrix.shape)
    y_test_np_matrix.tofile('y_test_np_matrix.dat')

   
    return [x_train_list, y_train_list, x_test_list, y_test_list]

def data_helper_no_list(tsharepdf,tetfpdf,day_frame):
    
    number_features=5
    pred_days=5 #predict future 5 days

    '''share'''
    
    #find how many days of the data
    StocksGroup=tsharepdf.groupby('No')
    Group1101=StocksGroup.get_group(1101)
    data_days=len(Group1101)

    x_train_list=[]
    x_test_list=[]
    AllFrameData=np.array()
    for index in range(data_days-(day_frame+pred_days)+1):
        DayFrameData=np.array()
        for stock_no,OneStockdf in StocksGroup:
            if len(OneStockdf) != data_days:
                continue
            OneStock_mx=OneStockdf.as_matrix()
            DayFrameData.append(OneStock_mx[index:index+day_frame,2:8]) #"No","Date","Name","Open","High","Low","Close","Volume"     
    AllFrameData.append(DayFrameData)    
                 
    number_train=round(0.5*AllFrameData.shape[0])
    #x train data
    x_train_list.append(AllFrameData[:int(number_train)])   
    
    #x test data
    x_test_list.append(AllFrameData[int(number_train):])
       
    '''ETF''' 
    ETFGroup=tetfpdf.groupby('No')
    
    y_train_list=[]
    y_test_list=[]

    for index in range(day_frame,OneETF_mx.shape[0]-pred_days+1):
        OneETFFrameData.append(OneETF_mx[index:index+pred_days,5]) #"No","Date","Name","Open","High","Low","Close","Volume"     

    for ETF_no,OneETFdf in ETFGroup:
        DayFrameData=np.array()
        OneETF_mx=OneETFdf.as_matrix()  
#         if ETF_no==690:
#             ETF_no=ETF_no
            
        if len(OneETFdf) < data_days:
            zero_m=np.zeros((data_days-len(OneETFdf),OneETF_mx.shape[1]))
            OneETF_mx=np.insert(arr=OneETF_mx, obj=0, values=zero_m, axis=0)
        OneETFFrameData=np.array(OneETFFrameData)
     
        OneETFFrameData=np.reshape(OneETFFrameData,(OneETFFrameData.shape[0],OneETFFrameData.shape[1],1))
        
        #y train data
        y_train_list.append(OneETFFrameData[:int(number_train)])                           
      
        #y test data
        y_test_list.append(OneETFFrameData[int(number_train):])

   
    return [x_train_list, y_train_list, x_test_list, y_test_list]

def build_model(input_length,input_dim):
    d=0.3
    model=Sequential()
    
    model.add(LSTM(256,input_shape=(input_length,input_dim),return_sequences=True))
    model.add(Dropout(d))
    
    model.add(LSTM(256,input_shape=(input_length,input_dim),return_sequences=False))
    model.add(Dropout(d))

    model.add(Dense(16,kernel_initializer="uniform",activation='relu'))
    model.add(Dense(5,kernel_initializer="uniform",activation='linear'))
    
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    
    return model

def denormalize(df,norm_value):
    original_value=df['Close'].values.reshape(-1,1)
    norm_value=norm_value.reshape(-1,1)
    
    min_max_scaler=preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm_value=min_max_scaler.inverse_transform(norm_value)
    
    return denorm_value

def TestGroup(dfs):
    groups=dfs.groupby('No')
    for no,nogroup in groups:
        print(no)
    g1=groups.get_group('1101')
    print(g1)
    for name,group in groups:
        print(group)
    print(dfs['1101'])                    
    print(dfs)
    print(dfs['Close'])
    
    close=dfs['Close']
    plt.plot(close)
    plt.show()
#
# Main entry point
#
def main(argv=None):
    tsharepdf=pd.read_csv('tsharepEnTitle.csv',encoding = 'big5',thousands=',',usecols=["No","Date","Open","High","Low","Close","Volume"],low_memory=False)  #nrows=100000,,verbose=True
    tetfpdf=pd.read_csv('tetfpEnTitle.csv',encoding = 'big5',thousands=',',usecols=["No","Date","Open","High","Low","Close","Volume"],low_memory=False)  
#     
#     
#     tsharepdf_norm=normalize(tsharepdf)
#     tetfpdf_norm=normalize(tetfpdf)
    
    
    x_train, y_train, x_test, y_test = data_helper(tsharepdf,tetfpdf, 20)
    
#     model=build_model(20,5)
#     model.fit(x_train, y_train, batch_size=128,epochs=10,validation_split=0.1,verbose=1)
    '''
    pred_train=model.predict(x_train)
    denorm_pred_train=denormalize(tsharepdf,pred_train)
    denorm_ytest_train=denormalize(tetfpdf,y_train)
    
    pred=model.predict(x_test)
    denorm_pred=denormalize(tsharepdf,pred)
    denorm_ytest=denormalize(tetfpdf,y_test)
    
    
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
'''
if __name__ == '__main__':
    main()