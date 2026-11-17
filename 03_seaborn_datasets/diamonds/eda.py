#diamonds

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#데이터
df = sns.load_dataset("diamonds")

print(df.head())
print(df.tail())
print(df.info())
print(df.describe(include="category"))
print(df['cut'].value_counts())
print(df['color'].value_counts())
print(df['clarity'].value_counts())
print(df.duplicated().sum())
print(df.isna().sum())


#전처리: 결축치, 중복, 인코딩
##결축치
df = df[(df["x"]>0) & (df["y"]>0) & (df["z"]>0)].reset_index(drop=True)

##인코딩
cut_order = ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']
color_order = ['G', 'E', 'F', 'H', 'D', 'I', 'J']
clarity_order = ['SI1', 'VS2', 'SI2', 'VS1', 'VVS2', 'VVS1', 'IF', 'I1']

df["cut_o"] = df["cut"].map({v:i for i,v in enumerate(cut_order)})
df["color_o"] = df["color"].map({v:i for i,v in enumerate(color_order)})
df["clarity_o"] = df["clarity"].map({v:i for i,v in enumerate(clarity_order)})

## x,y,z volume으로 묶기
df["volume"] = df["x"] * df["y"] * df["z"]


#이변량 분석
##수치형 price 별  carat/depth/table/volume
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
#
# sns.scatterplot(data=df, x="carat", y="price", ax=axes[0, 0])
# axes[0,0].set_title("price별 / carat", fontweight="bold")
#
# sns.scatterplot(data=df, x="depth", y="price", ax=axes[0, 1])
# axes[0,1].set_title("price별 / depth", fontweight="bold")
#
# sns.scatterplot(data=df, x="table", y="price", ax=axes[1, 0])
# axes[1,0].set_title("price별 / table", fontweight="bold")
#
# sns.scatterplot(data=df, x="volume", y="price", ax=axes[1, 1])
# axes[1,1].set_title("price별 / volume", fontweight="bold")
#
# plt.tight_layout()
# plt.show()

##범주형 price 별  cut/color/clarity
# fig, axes = plt.subplots(1,3, figsize=(15,10))
#
# sns.barplot(data=df, x="cut", y="price", ax=axes[0])
# axes[0].set_title("price별 / cut", fontweight="bold")
#
# sns.barplot(data=df, x="color", y="price", ax=axes[1])
# axes[1].set_title("price별 / color", fontweight="bold")
#
# sns.barplot(data=df, x="clarity", y="price", ax=axes[2])
# axes[2].set_title("price별 / clarity", fontweight="bold")
#
# plt.tight_layout()
# plt.show()


#상관관계
##수치형 price/carat/depth/table/volume
## -> price / carat, volume 상관관계가 가장 큼
fig, axes = plt.subplots(1,2, figsize=(15,10))

df_nums = df[["price", "carat", "depth", "table", "volume"]]
cor = df_nums.corr()
sns. heatmap(cor, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0])
axes[0].set_title("price/carat/depth/table/volume 상관관계", fontweight="bold")

plt.tight_layout()
plt.show()

