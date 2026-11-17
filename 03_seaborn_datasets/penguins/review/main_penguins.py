import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#데이터
df = sns.load_dataset("penguins")
df.to_csv("penguins.csv")

print(df.head())
print(df.tail())
print(df.info())
print(df.isna().sum())

#이변량이랑 #상관관계는 꼭 체크하고 model 시작해
