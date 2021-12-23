'''
Author: your name
Date: 2021-12-12 12:26:03
LastEditTime: 2021-12-12 12:27:43
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \irl\src\test.py
'''
from pettingzoo.mpe import simple_spread_v2

env = simple_spread_v2.env()


print(dir(env))

print(help(env))


