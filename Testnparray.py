import numpy as np

'''append test'''
# number_features=4
# 
# a=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
# b=[[10,20,30,40],[50,60,70,80],[90,100,110,120]]
# c=[[2,4,6,8],[10,12,14,16],[18,20,22,24]]
# 
# result=[]
# result.append(a)
# result.append(b)
# result.append(c)
# 
# result=np.array(result)
# 
# x_train=result[:3,:-1]
# print(x_train)
# 
# x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],number_features))
# print(x_train)

'''insert row test'''
a=np.array([[690, 20170331, 19.95, 19.9, 19.91, '      2,891'],
       [690, 20170405, 19.94, 19.89, 19.95, '        929'],
       [690, 20170406, 19.95, 19.9, 19.93, '        394']])

print(a)

print('======zero_m=======')
zero_m=np.zeros((2, 6))
print(zero_m)
# a=np.insert(a, 0, np.array((0, 0, 0, 0, 0, '        929')), 0) 
print('======insert=======')
a=np.insert(a, 0, zero_m, 0) 
print(a)

print('======contatenate======')
a=np.array([[1,2,3,4,5],
            [6,7,8,9,10],
            [11,12,13,14,15]])
b=np.array([[10,20,30,40,50],
            [60,70,80,90,100],
            [110,120,130,140,150]])

c=np.array([[11,22,33,44,55],
            [66,77,88,99,100],
            [111,122,133,144,155]])

print(a.shape)
print(b.shape)
print(a.shape)

train_list=[a,b,c]

# x_train_np_matrix=train_list[0]
# print(x_train_np_matrix)
# for i in range(1,len(train_list)):
#     x_train_np_matrix=np.concatenate((x_train_np_matrix,train_list[i]),axis=1)
#     print(x_train_np_matrix)
#     print(x_train_np_matrix.shape)
    
    
print('======contatenate multi once======')    
x_train_np_matrix=np.concatenate(train_list,axis=1)
print(x_train_np_matrix.shape)
print(x_train_np_matrix)
