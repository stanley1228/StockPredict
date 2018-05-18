import numpy as np

number_features=4

a=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
b=[[10,20,30,40],[50,60,70,80],[90,100,110,120]]
c=[[2,4,6,8],[10,12,14,16],[18,20,22,24]]

result=[]
result.append(a)
result.append(b)
result.append(c)

result=np.array(result)

x_train=result[:3,:-1]
print(x_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],number_features))
print(x_train)
