data = [
    ["古明地觉", "语文", 90],
    ["古明地觉", "数学", 95],
    ["古明地觉", "英语", 96],
    ["芙兰朵露", "语文", 87],
    ["芙兰朵露", "数学", 92],
    ["芙兰朵露", "英语", 98],
    ["琪露诺", "语文", 100],
    ["琪露诺", "数学", 9],
    ["琪露诺", "英语", 91],
]

import numpy as np
import pandas as pd

columns = ["姓名", "科目", "分数"]

data = np.array(data)
df = pd.DataFrame(data, columns=columns)

df = df.set_index(["姓名", "科目"])["分数"]
print(df)
df = df.unstack(level=0)
print(df)
# df = df.rename_axis(columns=None)
# print(df)
# df = df.reset_index()
# print(df)


pd.pivot(df, index="姓名", columns="科目", values="分数")
