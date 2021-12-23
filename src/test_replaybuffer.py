'''
Author: your name
Date: 2021-12-12 18:41:01
LastEditTime: 2021-12-12 19:00:29
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \irl\src\test_replaybuffer.py
'''
from tianshou.data import Batch
from tianshou.data import ReplayBuffer

buf = ReplayBuffer(2)
for i in range(2):
    print(buf.add(Batch({
        "obs": i, "act": i, "rew": i, "done": i
    })))

for i in range(2):
    print(buf.get(i, 'act'))

print(buf)

