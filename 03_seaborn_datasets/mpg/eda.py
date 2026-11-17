#mpg

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#데이터
df = sns.load_dataset("mpg")

print(df.head())
print(df.tail())
print(df.info())
print(df.isna().sum())


#결측치
##전처리
df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())

##인코딩
result = pd.get_dummies(df["origin"], drop_first=True)

##특징
features = df[["cylinders", "displacement", "horsepower", "weight"]]

##name -> brand
df["brand"] = df["name"].str.split().str[0]


#이변량
## y -> mpg
##x 축 수치형   mpg vs cylinders/displacement/horsepower/weight     / mpg vs model_year
# fig, axes = plt.subplots(2,2, figsize=(18,12))
#
# sns.boxplot(data=df, x="cylinders", y="mpg", ax=axes[0,0])
# axes[0,0].set_title("mpg / cylinders", fontweight="bold")
#
# sns.scatterplot(data=df, x="displacement", y="mpg", ax=axes[0,1])
# axes[0,1].set_title("mpg / displacement", fontweight="bold")
#
# sns.scatterplot(data=df, x="horsepower", y="mpg", ax=axes[1,0])
# axes[1,0].set_title("mpg / horsepower", fontweight="bold")
#
# sns.scatterplot(data=df, x="weight", y="mpg", ax=axes[1,1])
# axes[1,1].set_title("mpg / weight", fontweight="bold")

### mpg vs model_year
# fig, ax = plt.subplots(figsize=(10,5))
# sns.lineplot(data=df, x="model_year", y="mpg", ax=ax)
# ax.set_title("mpg / model_year", fontweight="bold")
#
# plt.tight_layout()
# plt.show()


##범주형   mpg vs origin/name
# fig, axes = plt.subplots(1,2, figsize=(15,10))
#
# sns.barplot(data=df, x="origin", y="mpg", ax=axes[0])
# axes[0].set_title("mpg / origin", fontweight="bold")
#
# sns. barplot(data=df, x="mpg", y="brand", ax=axes[1])
# axes[1].set_title("mpg / brand", fontweight="bold")
#
# plt.tight_layout()
# plt.show()


#상관관계
df_nums = df[["mpg", "cylinders", "displacement", "horsepower", "weight", "model_year"]]
cor = df_nums.corr()

fig, axes = plt.subplots(1,2, figsize=(15,10))

sns.heatmap(cor, cmap="coolwarm", annot=True, fmt=".2f", ax=axes[0])
axes[0].set_title("mpg/cylinders/displacement/horsepower/weight 상관관계", fontweight="bold")

##산점도   origin/name
sns.scatterplot(data=df, x="origin", y="name", hue="mpg", ax=axes[1])
axes[1].set_title("origin, name / mpg", fontweight="bold")

plt.tight_layout()
plt.show()
