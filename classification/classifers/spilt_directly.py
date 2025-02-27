import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import train_test_split

datas = pd.read_csv("xerces.csv")
train_set, test_set = train_test_split(datas, test_size=0.1, random_state=42)

train_set.to_csv("train_xerces_un.csv", index=False)
test_set.to_csv("test_xerces_un.csv", index=False)