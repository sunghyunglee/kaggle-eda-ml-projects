#diamonds

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#데이터
df = sns.load_dataset("diamonds")
df.to_csv("diamonds.csv")

print(df.head())
print(df.tail())
print(df.info())
print(df.isna().sum())
print(df.describe(include="category"))
print(df["cut"].value_counts())
print(df["color"].value_counts())
print(df["clarity"].value_counts())


#전처리
##결측치 <0으로 측정된 x,y,z 값을 처리 (결측치는 아니지만)>
###이건 x,y,z 각각 0인걸 걸러내기
# df = df[(df["x"]>0) & (df["y"]>0) & (df["z"]>0)].reset_index(drop=True)
###이건 x,y,z 전부 0으로 근데 0 아닌 것도 있으니 둘은 다름
# df = df[(df[["x","y","z"]] > 0)].reset_index(drop=True)
df = df[(df["x"]>0) & (df["y"]>0) & (df["z"]>0)]

##인코딩


#단변량, 이변량, 상관관계

##단변량, 이변량
fig, axes = plt.subplots(2,2, figsize=(20,12))

# sns.histplot(data=df["price"], bins= 20, ax=axes[0])
sns.histplot(data=df , x="price", bins= 20, ax=axes[0,0])
axes[0,0].set_title("Price", fontweight="bold")
sns.histplot(data=df, x="carat", bins= 20, ax=axes[0,1])
axes[0,1].set_title("Carat", fontweight="bold")
sns.scatterplot(data=df, x="carat", y="price", hue= "cut", ax=axes[1,0])
axes[1,0].set_title("Price / Carat", fontweight="bold")
sns.scatterplot(data=df, x="carat", y="price", hue= "clarity", ax=axes[1,1])
axes[1,1].set_title("Price / Carat", fontweight="bold")

plt.tight_layout()
plt.show()


##상관관계  <x 수치형>
df_nums = df[["price", "carat", "x", "y", "z"]]
cor = df_nums.corr()
print(cor)  #여기서 확인하고 수치형 다 넣어서 뺄 껀 빼고 heatmap 그리자

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(cor, annot=True, cmap="YlGnBu", fmt= ".2f", ax=ax)
ax.set_title("Price / Carat", fontweight="bold")

plt.tight_layout()
plt.show()


#인사이트
"""
x: 길이 (length, mm)
y: 너비 (width, mm)
z: 깊이 (depth, mm) — 테이블 면에서 큘릿(뾰족한 아랫부분)까지의 높이
x와 y는 값이 매우 비슷해요 (서로 상관관계가 굉장히 높음, 사실상 같은 정보에 가까움)
"""


