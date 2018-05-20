import pandas as pd

import numpy as np
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.recurrent  import LSTM
from keras.models import Model,load_model
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
#     print(x_train_np_matrix.shape)
    x_train_np_matrix.tofile('x_train_np_matrix.dat')
    
    x_test_np_matrix=np.concatenate(x_test_list,axis=2) 
#     print(x_test_np_matrix.shape)
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
            OneETFFrameData.append(OneETF_mx[index:index+pred_days,5].reshape(-1)) #"No","Date","Name","Open","High","Low","Close","Volume"     
        OneETFFrameData=np.array(OneETFFrameData)
     
#       OneETFFrameData=np.reshape(OneETFFrameData,(OneETFFrameData.shape[0],OneETFFrameData.shape[1],1))
        
        #y train data
        y_train_list.append(OneETFFrameData[:int(number_train)])                           
      
        #y test data
        y_test_list.append(OneETFFrameData[int(number_train):])

    y_train_np_matrix=np.concatenate(y_train_list,axis=1)
#     print(y_train_np_matrix.shape)
   
    y_test_np_matrix=np.concatenate(y_test_list,axis=1)
#     print(y_test_np_matrix.shape)
    np.savez('tain_test_data.npz',x_train_np_matrix,x_test_np_matrix,y_train_np_matrix,y_test_np_matrix)
    #d=np.load('tain_test_data.npz')
    #x_train_np_matrix=d['arr_0']
    #x_test_np_matrix==d['arr_1']
    #y_train_np_matrix=d['arr_2']
    #y_test_np_matrix==d['arr_3']
    
    return [x_train_np_matrix, y_train_np_matrix, x_test_np_matrix, y_test_np_matrix]


def build_model(input_length,input_dim,output_dim):
    d=0.3
    model=Sequential()
    
    model.add(LSTM(256,input_shape=(input_length,input_dim),return_sequences=True))
    model.add(Dropout(d))
    
    model.add(LSTM(256,input_shape=(input_length,input_dim),return_sequences=False))
    model.add(Dropout(d))

    model.add(Dense(16,kernel_initializer="uniform",activation='relu'))
    model.add(Dense(output_dim,kernel_initializer="uniform",activation='linear'))
    
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
    
def TrainProcess():
    tsharepdf=pd.read_csv('tsharepEnTitle.csv',encoding = 'big5',thousands=',',usecols=["No","Date","Open","High","Low","Close","Volume"],low_memory=False)  #nrows=100000,,verbose=True
    tetfpdf=pd.read_csv('tetfpEnTitle.csv',encoding = 'big5',thousands=',',usecols=["No","Date","Open","High","Low","Close","Volume"],low_memory=False)  
#         
#         
#     tsharepdf_norm=normalize(tsharepdf)
#     tetfpdf_norm=normalize(tetfpdf)
#        
#        
#     x_train, y_train, x_test, y_test = data_helper(tsharepdf_norm,tetfpdf_norm, 20)
    
    d=np.load('train_test_data.npz')
    x_train=d['arr_0']
    print(x_train.shape)
    x_test=d['arr_1']
    print(x_test.shape)
    y_train=d['arr_2']
    print(y_train.shape)
    y_test=d['arr_3']
    print(y_test.shape)
      
    '''train model'''
#     model=build_model(20,x_train.shape[2],y_train.shape[1])
#     model.fit(x_train, y_train, batch_size=128,epochs=50,validation_split=0.1,verbose=1)
#     print('fit done')
    
#    '''load model'''
    print('load stock_model_batch256.h5...')
    model = load_model('stock_model_batch256.h5')
    print('load stock_model_batch1256.h5...Done')
    
#     model.save('stock_model_batch128.h5')
#     print('model.save')

   
    pred_y_train=model.predict(x_train)
    print('predict done')
    
    denorm_pred_y_train=denormalize(tetfpdf,pred_y_train)
    denorm_y_train=denormalize(tetfpdf,y_train)
    print(denorm_y_train.shape)

    denorm_y_train=np.reshape(denorm_y_train,(y_train.shape[0],-1,5))
    denorm_pred_y_train=np.reshape(denorm_pred_y_train,(y_train.shape[0],-1,5))
    print(denorm_y_train.shape)
    
    pred_y_test=model.predict(x_test)
    denorm_pred_y_test=denormalize(tetfpdf,pred_y_test)
    denorm_y_test=denormalize(tetfpdf,y_test)
    denorm_pred_y_test=np.reshape(denorm_pred_y_test,(y_test.shape[0],-1,5))
    denorm_y_test=np.reshape(denorm_y_test,(y_test.shape[0],-1,5))
    
    
    '''
    show train data result
    '''
    '''plot by day'''
    plt.figure()
    plt.plot(denorm_pred_y_train[:,:,0],color='red',label='Prediction')
    plt.plot(denorm_y_train[:,:,0],color='blue',label='Answer')
    plt.title('train result predict next 1 day')
    '''plot by company'''
    plt.figure()
    plt.plot(denorm_pred_y_train[:,0,:],color='red',label='Prediction')
    plt.plot(denorm_y_train[:,0,:],color='blue',label='Answer')
    plt.title('train result predict y train on of 1th company')
    plt.legend(loc='best')
    
    '''
    show test data result
    '''
    '''plot by day'''
    plt.figure()
    plt.plot(denorm_pred_y_test[:,:,0],color='red',label='Prediction')
    plt.plot(denorm_y_test[:,:,0],color='blue',label='Answer')
    plt.title('test result predict next 1 day')
    '''plot by company'''
    plt.figure()
    plt.plot(denorm_pred_y_test[:,1,:],color='red',label='Prediction')
    plt.plot(denorm_y_test[:,1,:],color='blue',label='Answer')
    plt.title('test result predict y test of 1th company')
    plt.legend(loc='best')
    plt.show()
    
    #print(tsmcdf_norm)
    #close=tsmcdf['Close']
    #close.plot()
    #plt.plot(close)
    #plt.show()

#
# Main entry point
#
def main(argv=None):
    #Testpandas()
    TrainProcess()
    

if __name__ == '__main__':
    main()
    