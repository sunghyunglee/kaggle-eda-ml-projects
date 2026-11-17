#mpg

import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#데이터
df = sns.load_dataset("mpg")
df.to_csv("data/mpg.csv")

print(df.head())
print(df.tail())
print(df.info())
print(df.duplicated().sum())
print(df.isna().sum())

## -> mpg 결정하는 주요 요인 찾기 / category 없고, 결측치 1(6)
'''
Cylinders 기통 수
Displacement 배기량
Horsepower 마력
'''


#전처리: 결측치, 중복, 인코딩
##결측치
df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())

## name -> brand로
df["brand"] = df["name"].str.split().str[0]


#단변량


#이변량
##y -> mpg
## 수치형 x -> cylinders/displacement/horsepower/weight/acceleration/model_year
## 범주형 x -> origin/name
### mpg 별 수치형 x
# fig, axes = plt.subplots(2,3, figsize=(18,12))
#
# sns.boxplot(data= df, x="cylinders", y="mpg", ax=axes[0,0])
# axes[0,0].set_title("mpg / cylinders", fontweight="bold")
#
# sns.scatterplot(data= df, x="displacement", y="mpg", ax=axes[0,1])
# axes[0,1].set_title("mpg / displacement", fontweight="bold")
#
# sns.scatterplot(data= df, x="horsepower", y="mpg", ax=axes[0,2])
# axes[0,2].set_title("mpg / horsepower", fontweight="bold")
#
# sns.scatterplot(data= df, x="weight", y="mpg", ax=axes[1,0])
# axes[1,0].set_title("mpg / weight", fontweight="bold")
#
# sns.scatterplot(data= df, x="acceleration", y="mpg", ax=axes[1,1])
# axes[1,1].set_title("mpg / acceleration", fontweight="bold")
#
# sns.boxplot(data= df, x="model_year", y="mpg", ax=axes[1,2])
# axes[1,2].set_title("mpg / model_year", fontweight="bold")
#
# plt.tight_layout()
# plt.show()

### mpg 별 범주형 x (origin/name)
# fig, axes = plt.subplots(1,2, figsize=(25,15))
# # sns.barplot(data= df, x="origin", y="mpg", ax=axes[0])
# axes[0].set_title("mpg / origin", fontweight="bold")
# # sns.barplot(data= df, x="brand", y="mpg", ax=axes[1])
# axes[1].set_title("mpg / brand", fontweight="bold")
#
# fig, ax = plt.subplots(figsize=(10, 5))
# sns.barplot(data= df, x="origin", y="mpg")
# ax.set_title("mpg / origin")
#
# plt.tight_layout()
# plt.show()
#
# fig, ax = plt.subplots(figsize=(25, 12))
# sns.barplot(data= df, x="brand", y="mpg")
# ax.set_title("mpg / brand")
#
# plt.tight_layout()
# plt.show()


#상관관계
##수치형 mpg/cylinders/displacement/horsepower/weight/acceleration/model_year
### mpg -> model_year/acceleration
fig, axes = plt.subplots(1,2, figsize=(20, 10))
df_nums = df[["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]]
cor = df_nums.corr()
sns.heatmap(cor, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0])
axes[0].set_title("mpg/cylinders/displacement/horsepower/weight/acceleration/model_year", fontweight="bold")

##범주형 origin/name(brand)
sns.boxplot(data=df, x="origin", y="brand", ax=axes[1])
axes[1].set_title("origin/brand", fontweight="bold")

plt.tight_layout()
plt.show()

##mpg/model_year/acceleration 추려서 lineplot 해보기
# fig, axes = plt.subplots(1,2, figsize=(20, 10))
# sns.lineplot(data=df, x="model_year", y="mpg", ax=axes[0])
# axes[0].set_title("mpg/model_year", fontweight="bold")
#
# sns.lineplot(data=df, x="acceleration", y="mpg", ax=axes[1])
# axes[1].set_title("mpg/acceleration", fontweight="bold")
#
# plt.tight_layout()
# plt.show()


#인사이트
'''
mpg / model_year 주목해서 해보고
acceleration은 관찰할만 하다 (sklearn으로 model 여부에 넣을지는 생각해봐야겠다)
-> 잘못 분석함 // model_year가 올라갈수록 mpg가 좋아지는 추세정도고 상관관계가 높지 않다.
mpg에 가장 큰 영향을 주는 건 weight (-0.83), displacement (-0.80), cylinders (-0.78)
horsepower (-0.77)로 주목할만 함
    -> 양,음이 중요한게 아니라 절대값이 중요하고 색이 진한지가 중요한
        진할수록 상관관계가 높음


origin/name
name을 brand로 묶어서 진행했고
일본 브랜드 - 유럽 브랜드 - 미국 브랜드 순으로 높았다
sklearn으로 model 여부에 넣을지는 생각해봐야겠다
'''

