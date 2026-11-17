#flights    -> 시계열 데이터 (순서유지 해야됨)
'''
시계열 데이터
    추세, 계절성(성수기/비성수기, 아니면 진짜 계절 봄,여름,가을,겨울)
    ex) 자연재해, 코로나
'''

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

df = sns.load_dataset(("flights"))
df.to_csv("data/flights.csv")

# print(df.head())
# print(df.tail())
# print(df.info())
# print(df.describe(include="category"))
# print(df.isna().sum())

#날짜로 바꾸기
x = df["year"].astype(str) + "-" + df["month"].astype(str)
df["date"] = pd.to_datetime(x, format="%Y-%b")
df = df.sort_values('date').reset_index(drop=True)

print(df)


#그래프
# plt.figure(figsize=(10, 5))
# plt.plot(df["date"], df["passengers"])
#
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(data=df, x="month", y="passengers")

plt.tight_layout()
plt.show()

