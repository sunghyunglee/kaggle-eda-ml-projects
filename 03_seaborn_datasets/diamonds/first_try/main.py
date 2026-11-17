#diamonds

import seaborn as sns
import numpy as np
import sklearn
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

df = sns.load_dataset("diamonds")
df.to_csv("data/diamonds.csv")

# print(df.head())
# print(df.tail())
# print(df.info())
# print(df.describe(include="category"))
# print(df.isna().sum())


#단변량 분석
##y값 분석
print(df["price"].value_counts())
##x값 분석
df["volume"] = df["x"] * df["y"] * df["z"]
##1. 숫자형 (depth, table, carat / x,y,z = volume)
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#
# sns.histplot(df["depth"], bins=20, ax=axes[0, 0])
# axes[0, 0].set_title("Depth", fontweight="bold")
#
# sns.histplot(df["table"], bins=20, ax=axes[0, 1])
# axes[0, 1].set_title("Table", fontweight="bold")
#
# sns.histplot(df["carat"], bins=20, ax=axes[1, 0])
# axes[1, 0].set_title("Carat", fontweight="bold")
#
# sns.histplot(df["volume"], bins=20, ax=axes[1, 1])
# axes[1, 1].set_title("Volume", fontweight="bold")
#
#
# plt.tight_layout()
# plt.show()

##범주형 (cut, color, clarity)
# fig,axes = plt.subplots(1, 3, figsize=(15, 10))
#
# sns.countplot(data=df, x="cut", ax=axes[0])
# axes[0].set_title("Cut", fontweight="bold")
#
# sns.countplot(data=df, x="color", ax=axes[1])
# axes[1].set_title("Color", fontweight="bold")
#
# sns.countplot(data=df, x="clarity", ax=axes[2])
# axes[2].set_title("Clarity", fontweight="bold")
#
# plt.tight_layout()
# plt.show()


#이변량 분석
##price별 / (depth, table, carat, / x,y,z = volume)
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#
# sns.scatterplot(data=df, x="depth", y="price", ax=axes[0,0])
# axes[0,0].set_title("Price / Depth", fontweight="bold")
#
# sns.scatterplot(data=df, x="table", y="price", ax=axes[0,1])
# axes[0,1].set_title("Price / Table", fontweight="bold")
#
# sns.scatterplot(data=df, x="carat", y="price", ax=axes[1,0])
# axes[1,0].set_title("Carat / Price", fontweight="bold")
#
# sns.scatterplot(data=df, x="volume", y="price", ax=axes[1,1])
# axes[1,1].set_title("Price / Volume", fontweight="bold")
#
# plt.tight_layout()
# plt.show()

##pricet별 / (cut, color, clarity)
# fig, axes = plt.subplots(1, 3, figsize=(15, 10))
#
# sns.barplot(data=df, x="cut", y="price", ax=axes[0])
# axes[0].set_title("Price / Cut", fontweight="bold")
#
# sns.barplot(data=df, x="color", y="price", ax=axes[1])
# axes[1].set_title("Price / Color", fontweight="bold")
#
# sns.barplot(data=df, x="clarity", y="price", ax=axes[2])
# axes[2].set_title("Price / Clarity", fontweight="bold")
#
# plt.tight_layout()
# plt.show()


#상관관계 depth, table, carat / x,y,z = volume
fig, axes = plt.subplots(1,2, figsize=(15,10))
df_nums = df[["depth", "table" , "carat", "price", "volume"]]
cor = df_nums.corr()
sns.heatmap(cor, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0])
axes[0].set_title("depth, table, carat, volume 상관관계", fontweight="bold")

##산점도 carat / price hue -> volume or cut
sns.scatterplot(data=df, x="carat", y="price", ax=axes[1], hue="cut")
axes[1].set_title("Carat / Price", fontweight="bold")

plt.tight_layout()
plt.show()


#cut_order 만들기
cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
num_cols = ["depth", "table" , "carat", "price", "volume"]
result = df[num_cols].corr()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(data=df, x="cut", y="price", order=cut_order, ax=axes[0])
sns.boxplot(data=df, x="clarity", y="price", ax=axes[1])

plt.tight_layout()
plt.show()


#인사이트
'''
price / carat, volume (x,y,z) 가 중요함

cut / color / clarity 교란변수
    "Fair"(가장 낮은 등급의 컷)가 오히려 "Ideal"(가장 좋은 컷)보다 평균 가격이 높게 나옴
'''

