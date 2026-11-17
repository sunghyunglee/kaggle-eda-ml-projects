import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
EDA - (Exploratory Data Analysis, 탐색적 데이터 분석)는
본격적인 데이터 분석이나 머신러닝 모델링에 들어가기 전,
데이터를 다양한 각도에서 살펴보고 특징을 파악하는 초기 과정
구조 -> 결측치 -> 변수, 관계
'''


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False



#데이터로드
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

#데이터파악
# print(df.head())
# print(df.info())
# print(df.isna().sum())



#결측지 처리 (Age, Cabin, Embarked)
# print(df.dropna())
df["Age"] = df["Age"].fillna(df["Age"].median())
df.drop(columns=["Cabin"])
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])



#단변량 분석
# ##생존사 사망자 비율
# print(df["Survived"].value_counts(normalize=True) *100)
#
# ##범주형 시각화
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].set_title("생존여부")
# sns.countplot(x="Survived" , data=df, ax=axes[0])
#
# axes[1].set_title("성별")
# sns.countplot(x="Sex" , data=df, ax=axes[1])
#
# axes[2].set_title("객실 등급")
# sns.countplot(x="Pclass" , data=df, ax=axes[2])
#
# ##나이 시각화
# plt.figure(figsize=(5, 5))
# sns.histplot(df["Age"] , bins=30, kde=True)
# plt.title("나이")
#
# plt.show()



#이변량 분석 (성별, 객실등급, 나이대, 가족 수에 따른 생존율 파악)
# fig, axes = plt.subplots(2, 2, figsize=(12,10))
#
# sns.barplot(x="Sex" , y="Survived" , data=df, ax=axes[0,0])
# sns.barplot(x="Pclass" , y="Survived" , data=df, ax=axes[0,1])
#
# ##feature engineering   *새로운 정보를 생성해서 결과값을 만듬   <pd.cut()>
# df["AgeBand"] = pd.cut(
#     df["Age"],
#     bins=[0,12,18,40,60,100],
#     labels=["아동", "청소년", "청년", "중년", "노년"],
# )
# print(df.groupby("AgeBand")["Survived"].mean())
# sns.barplot(x="AgeBand" , y="Survived" , data=df, ax=axes[1,0])
#
# df["FamilySize"] = df["SibSp"] + df["Parch"] +1
# df["FamilyBand"] = pd.cut(
#     df["FamilySize"],
#     bins=[0, 1, 3, 5, 7, 9],
#     labels=["싱글", "소가족", "중가족", "대가족", "슈퍼가족"],
# )
# print(df.groupby("FamilySize")["Survived"].mean())
# sns.barplot(x="FamilySize" , y="Survived" , data=df, ax=axes[1,1])    #상대적으로 정확하지 않음
# sns.barplot(x="FamilyBand" , y="Survived" , data=df, ax=axes[1,1])



#상관관계
# cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
# cor = df[cols].corr()
#
# plt.figure(figsize = (10,8))
# sns.heatmap(cor, cmap="coolwarm", annot=True, fmt=".2f")
#
# plt.show()



#인사이트
'''
여성의 생존율이 남성보다 높다
객실 등급이 좋을수록 (숫자가 낮을수록) 생존율이 높다
나이가 어린 아동의 경우 생존율이 높다
가족 규모가 2~4인 일 때 생존율이 높다
'''



#이변량 분석2    <활용>
fig, axes = plt.subplots(2, 2, figsize=(20,10))

##성별 + 객실등급에 따른 생존율 파악
# df["SPclass"] = df["Sex"] + df["Pclass"].astype("str")
# print(df.groupby("SPclass")["Survived"].mean())
# # sns.barplot(x="SPclass" , y="Survived" , data=df, ax=axes[0,0])
# sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df, ax=axes[0,0])

result = df.groupby(["Sex" , "Pclass"])["Survived"].mean().unstack()    #unstack - 깔끔하게 정리하기 좋음, 히트맵에 좋음
sns.heatmap(result, cmap="coolwarm", annot=True, fmt=".2f", ax=axes[0,0])
# plt.title("성별 + 객실등급에 따른 생존율 파악")
axes[0,0].set_title("성별 + 객실등급에 따른 생존율 파악", weight="bold")


##탑승 항구에 따른 생존율 파악  <교란변수>
sns.barplot(x="Embarked", y="Survived" , data=df, ax=axes[0,1])
# plt.title("탑승 항구에 따른 생존율 파악")
axes[0,1].set_title("탑승 항구에 따른 생존율 파악", weight="bold")

result02 = pd.crosstab(df["Embarked"], df["Pclass"])    #groupby로 할때랑 결과가 같음


##이름에 따른 생존율 파악
'''
정규 표현식 (Regular Expression)     .str.extract(r"") <pandas 기능>
    .str.extract(r",\s*([^\.]+)\.")
'''
# df["Name"].str.extract(r',\s*([^\.]+)\.')
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")     #mr/mrs/miss 등 찾기 용이함
x = ["Mr", "Miss", "Mrs", "Master"]
df["Title"] = df["Title"].where(df["Title"].isin(x), "Rare")
# print(df["Title"].value_counts())
# result03 = df.groupby("Title")["Survived"].mean()
# print(result04)
sns.barplot(x="Title" , y="Survived" , data=df, ax=axes[1,0])
# plt.title("이름에 따른 생존율 파악")
axes[1,0].set_title("이름에 따른 생존율 파악", weight="bold")


##운임에 따르 생존율 파악  <Fare>
# df["FareBand"] = pd.cut(
#     df["Fare"],
#     bins=[0, 10, 100, 300, 600],
#     labels=["저가", "보통", "고가", "vip"]
# )
df["FareBand"] = pd.qcut(df["Fare"], 4, labels=["저가", "중가", "고가" , "vip"])    #qcut - 균등하게 쪼개기 좋음
sns.barplot(x="FareBand" , y="Survived" , data=df, ax=axes[1,1])
# plt.title("운임에 따르 생존율 파악")
axes[1,1].set_title("운임에 따른 생존율 파악", weight="bold")

# result04 = df.groupby("FareBand")["Survived"].mean()


#출력
plt.tight_layout()
plt.show()


#인사이트
'''
##성별 + 객실등급에 따른 생존율 파악  (Age + Pclass)
여성의 생존율은 모든 등급에서 남성보다 압도적으로 높으며, 그 격차는 등급이 높을수록(1등급) 더 크게 벌어진다

##탑승 항구에 따른 생존율 파악  (Embarked) <교란변수>
C = 셰르부르, Q=퀸즈타운
C의 높은 생존율은 항구 자체의 효과라기보다, C 탑승객 중 1등급 비율이 높았던 것이 실제 원인일 가능성이 크다.

##이름에 따른 생존율 파악 (Name)
Mr의 생존율이 가장 낮다 (성인 남성 다수를 차지하는 그룹)
Master는 Mr보다 훨씬 높아, 남성이라도 소년은 우선 구조 대상이였다.
분석 결과 (Dr, Col, Mme, Ms, Lady 등)의 절반가량이 표본 수가 1~2명뿐이라 통계적으로 신뢰할 수 없다.

##운임에 따르 생존율 파악 (Fare)
운임(Fare)이 높을수록 생존율이 단계적으로 상승하며, 이는 Pclass와 관련이 있다.
'''

