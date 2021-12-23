# 算法流程

1. 创建一个环境
    1. 获取环境的`状态空间`的大小
    2. 获取环境的`动作空间`的大小
2. 根据获取的`状态空间`和`动作空间`信息分别作为输入空间和输出空间构建网络结构
3. 创建一个优化器，用于优化神经网络的参数
4. 根据如上创建的网络结构和优化器创建一个`Policy`
5. 创建一个数据缓冲区，由于存储`Collector`从环境搜集的数据
6. 创建一个`Collector`
7. 利用`onpolicy`和`offpolicy`来训练数据

![img_1.png](img_1.png)

如上流程图执行顺序为：
1. collector.collect()
2. policy()
3. model()
4. env.step()
5. buffer.add()
6. policy.update()

`buffer`中的一个数据项为：
```python
import numpy as np
from tianshou.data import Batch

d = Batch(
    obs=np.array([[0.11245313, 0.6052373, -0.15940769, -1.1484663]], dtype=np.float32),
    act=np.array([1]),
    rew=np.array([1]),
    done=np.array([False]),
    obs_next=np.array([[0.12455787, 0.412514, -0.18237701, -0.90971416]], dtype=np.float32),
    info={},
    policy=Batch(),
)
```
