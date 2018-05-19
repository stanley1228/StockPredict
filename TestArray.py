import numpy as np

a_day1=[32,38,31,38,100]
a_day2=[33,39,32,39,133]
a_day3=[34,40,33,32,122]
a_day4=[36,42,37,38,206]

mq_day1=[100,105,92,100,1000]
mq_day2=[99,110,90,100,1005]
mq_day3=[103,110,100,100,10]
mq_day4=[102,108,95,101,2000]

a_OneStock_mx=np.array([a_day1,a_day2,a_day3,a_day4])

mq_OneStock_mx=np.array([mq_day1,mq_day2,mq_day3,mq_day4])

AllStock_mx=[a_OneStock_mx,mq_OneStock_mx]
print('==============AllStock_mx==============')
print(AllStock_mx)

data_days=4
day_frame=2
pred_days=1


AllStockFrameData_list=[]
x_train_list=[]
x_test_list=[]
for OneStock_mx in AllStock_mx:
    OneStockFrameData=[]
    for index in range(data_days-(day_frame+pred_days)+1):
        OneStockFrameData.append(OneStock_mx[index:index+day_frame,2:5]) #"No","Date","Name","Open","High","Low","Close","Volume"  
    print('===============OneStockFrameData list===============')
    print(OneStockFrameData)
    

    print('===============OneStockFrameData_np_array===============')
    OneStockFrameData=np.array(OneStockFrameData)
    print(OneStockFrameData)
      
    OneStockFrameData=np.reshape(OneStockFrameData,(OneStockFrameData.shape[0],OneStockFrameData.shape[1],3))
    print('===========OneStockFrameData_np_array_reshape===========')
    print(OneStockFrameData)

    print('===========number_train===========')
    number_train=round(0.5*OneStockFrameData[0].shape[0])
    print('number_train=')
    print(number_train)
    
    x_train_list.append(OneStockFrameData[:int(number_train)])
    x_test_list.append(OneStockFrameData[int(number_train):])
  



print('===========x_train_list===========')
print(x_train_list)
print('===========x_test_list===========')
print(x_test_list)

 
