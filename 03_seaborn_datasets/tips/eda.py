#tips

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#데이터
df=sns.load_dataset("tips")

print(df.head())
# print(df.describe())
# print(df.info())


#결측치, 중복제거, 인코딩
# print(df.isna().sum())
# print(df.isna().mean())
##결측치 없음


#단변량 분석
##y값 분석
print(df["tip"].value_counts())
##x값 분석
#1. x 숫자형 (total_bill, size)
# fig, axes = plt.subplots(1,2, figsize=(8,6))
#
# sns.histplot(df["total_bill"], bins= 20, ax=axes[0])
# axes[0].set_title("Total Bill", fontweight="bold")
#
# sns.histplot(df["size"], bins=6, ax=axes[1])
# axes[1].set_title("Size", fontweight="bold")
#
# plt.tight_layout()
# plt.show()

##2. x 범주형 (sex, smoker, day, time)
# fig, axes = plt.subplots(2, 2, figsize=(10,8))
#
# sns.countplot(data=df, x="sex", ax=axes[0,0])
# axes[0,1].set_title("Sex", fontweight="bold")
#
# sns.countplot(data=df, x="smoker", ax=axes[0,1])
# axes[0,1].set_title("Smoker", fontweight="bold")
#
# sns.countplot(data=df, x="day", ax=axes[1,0])
# axes[1,0].set_title("Day", fontweight="bold")
#
# sns.countplot(data=df, x="time", ax=axes[1,1])
# axes[1,1].set_title("Time", fontweight="bold")
#
# plt.tight_layout()
# plt.show()


#이변량 분석
##팁별 / total_bill, 팁별 / size
print(df.groupby("tip")["total_bill"].mean())
print(df.groupby("tip")["size"].mean())

#boxplot 이상하게 나와서 산점도로 함    <또 다른 다른걸로 함>
# fig, axes = plt.subplots(2,1, figsize=(15,10))
# sns.boxplot(data=df, x="tip", y="total_bill", ax=axes[0])
# axes[0].set_title("tip / total_bill", fontweight="bold")
# sns.boxplot(data=df, x="tip", y="size", ax=axes[1])
# axes[1].set_title("tip / size", fontweight="bold")


##팁별 / total_bill -> regplot, 팁별 / size -> barplot
# fig, axes = plt.subplots(1,2, figsize=(15,8))
#
# sns.regplot(data=df, x="total_bill", y="tip", ax=axes[0])
# axes[0].set_title("팁별 / total_bill", fontweight="bold")
# sns.barplot(data=df, x="size", y="tip", ax=axes[1])
# axes[1].set_title("팁별 / size", fontweight="bold")
#
# plt.tight_layout()
# plt.show()


##팁별 / sex, 팁별 / smoker, 팁별 / day, 팁별 / time
# fig, axes = plt.subplots(2,2, figsize=(15, 10))
#
# sns.barplot(data=df, x="sex", y="tip", ax=axes[0,0])
# axes[0,0].set_title("팁별 / 성별", fontweight="bold")
#
# sns.barplot(data=df, x="smoker", y="tip", ax=axes[0,1])
# axes[0,1].set_title("팁별 / 흡연", fontweight="bold")
#
# sns.barplot(data=df, x="day", y="tip", ax=axes[1,0])
# axes[1,0].set_title("팁별 / 날", fontweight="bold")
#
# sns.barplot(data=df, x="time", y="tip", ax=axes[1,1])
# axes[1,1].set_title("팁별 / 시간", fontweight="bold")
#
# plt.tight_layout()
# plt.show()


#상관관계
# ## tip / total_bill / size
# plt.figure(figsize=(10, 8))
# df_nums = df[["tip", "total_bill", "size"]]
# cor = df_nums.corr()
# sns.heatmap(cor, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("tip / total_bill / size 상관관계")
#
# plt.tight_layout()
# plt.show()

##산점도: total_bill / tip
plt.figure(figsize=(10, 8))

sns.scatterplot(data=df, x="total_bill", y="tip", hue="sex")
plt.title("total_bill / tip")

plt.tight_layout()
plt.show()


#인사이트
'''
#단변량 분석
1. x 숫자형 (total_bill, size)
total_bill
    10이상 25미만에서 높고
    10~20 사이에 최대치 그래프가 있다 (35이상)
    40초과해서는 전체적으로 낮다
size
    size 2가 가장 높고 3,4 가 그 다음으로 높다

2. x 범주형
sex - 남성이 여성보다 많다 (대략 150 / 90)
smoker - 비흡연자가 더 많다 (대략 90/ 150)
day - 목,금,토,일 중에서 금요일이 가장 적고 (20), 토요일이 (대략 80이상으로 가장 많다)
    주말이 평일보다 많다
time - 저녁이 175로 압도적으로 점심 (대략 65)보다 많다


#이변량 분석
##팁별 / total_bill -> regplot, 팁별 / size -> barplot
    팁별 / tootal_bill
        팁별 2~4에서 total_bill 10~20 사이에 가장 몰려 있음
    팁별 / size
        사이즈가 높을 수록 팁이 높아지고 가장 큰 사이즈 6에서 팁이 가장 높다
        단, 사이즈 5는 4보다 팁이 적다
##팁별 / sex, 팁별 / smoker, 팁별 / day, 팁별 / time
    팁별 / sex
        남성이 3이상으로 2.7~8 정도의 여성보다 높다 <남성>여성>
    팁별 / smoker
        흡연자가 팁을 조금 더 주지만 수치화 할 정도로 차이가 나지 않고 중요하지 않다
    팁별 / day
        주말이 평일보다 팁이 높으며 일요일이 가장 높다 <금요일은 적은 인원수에 비해 팁이 높다>
    팁별 / time
        저녁이 점심보다 팁이 높다 <수치화 할 정도로 엄청난 차이는 아니다>
        
        
#상관관계
## tip / total_bill / size
셋의 값이 모두 양의 값을 띄면서 셋의 상관관계가 좋다? 높다?
    tip / size
        0.49 상관관게고
    tip / total_bill
        0.68
    total_bill / size
        0.6


#산점도
##total_bill / tip (hue=sex)
    total_bill 10~20 구간에 tip 2~4 사이가 가장 밀집되어 있음
    total_bill이 커질수록 tip도 대체로 같이 증가하는 우상향 패턴 (양의 상관관계)
    total_bill 40 이상 구간은 데이터 수 자체가 적어서 점이 듬성듬성함
'''
