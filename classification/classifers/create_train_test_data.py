# 分割数据集
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random


train_size = 0.9 # 90%数据用来训练，10%数据用来测试

datas = pd.read_csv("xerces.csv")
index = np.where(datas.values[:, -1] == "Y")[0]

train_index, test_index = train_test_split(index, train_size=train_size)

o_index = np.where(datas.values[:, -1] == "N")[0]

o_index = random.sample(list(o_index), test_index.shape[0])

test_index = o_index + list(test_index)
train_index = [i for i in range(datas.shape[0])]
for i in test_index:
    train_index.remove(i)

test_data = datas.loc[test_index, :]
train_data = datas.loc[train_index, :]

train_data.to_csv("train_xerces.csv", index=False)
test_data.to_csv("test_xerces.csv", index=False)
