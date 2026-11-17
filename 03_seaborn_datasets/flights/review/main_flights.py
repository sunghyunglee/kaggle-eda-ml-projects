#flights

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#데이터
df = sns.load_dataset("flights")
df.to_csv("flights.csv")

print(df.head())
print(df.tail())
print(df.info())
print(df.isna().sum())
print(df.describe(include='category'))
print(df["month"].value_counts())


#전처리
df["date"] = pd.to_datetime(
    df["year"].astype(str) + "-" + df["month"].astype(str),
    format="%Y-%b",
)

df = df.sort_values("date").reset_index(drop=True)


#단변량, 이변량, 상관관계

