import pandas as pd
import numpy as np
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                              'foo', 'bar', 'foo', 'foo'],
                       'B' : ['one', 'one', 'two', 'three',
                              'two', 'two', 'one', 'three'],
                       'C' : np.random.randn(8),
                       'D' : np.random.randn(8)})
print(df)

groups=df.groupby('A')
print(len(groups))
list_group=list(groups)
print(len(list_group[0]))



# # for qqq,name123 in groups:
# #     print(qqq)
# #     print(name123)
# for name,group in groups:
#     print(len(group))
#     print(name)
#     data=group.as_matrix()
#     print(data)
#     
# print('\n')
# print(data[1:3,:3])