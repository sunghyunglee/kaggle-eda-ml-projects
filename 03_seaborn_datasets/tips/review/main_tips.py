import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#데이터
df = sns.load_dataset("tips")
df.to_csv("tips.csv")

print(df.head())
print(df.tail())
print(df.info())
print(df.isna().sum())
print(df.describe(include="category"))
print(df["sex"].value_counts())
print(df["smoker"].value_counts())
print(df["day"].value_counts())
print(df["time"].value_counts())

# 전처리 -> 여기 결측치 없음
##인코딩
'''
이미 다 합쳐진 상태라서 features 만들고 concat 안해도 됨
'''
# df = pd.get_dummies(df, columns = ["sex", "smoker", "day", "time"])

##파생 변수
df["tip_pct"] = df["tip"] / df["total_bill"] * 100
print(df["tip_pct"].describe())

'''
features = ["total_bill", "size"]
X = df[features]
y = df["tip"]
이렇게 안써도 됨
'''
# 이렇게 간단하게 됨
df = pd.get_dummies(df, columns=["sex", "smoker", "day", "time"], drop_first=True)

X = df.drop(columns=["tip"])
y = df["tip"]



