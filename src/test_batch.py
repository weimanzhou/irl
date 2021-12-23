'''
Author: your name
Date: 2021-12-12 17:22:35
LastEditTime: 2021-12-12 18:39:05
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \irl\src\test_batch.py
'''
from tianshou.data import Batch


d1 = Batch(c=Batch(d=1))

d1['a'] = 1
d1.b = 2

# print(d1[0])
print(d1.b)
print(d1['a'])

print(d1)
