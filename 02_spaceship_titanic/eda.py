#Spaceship Titanic

import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

# Spaceship_Titanic = sns.load_dataset("Spaceship Titanic") #seaborn에 없어서 이렇게 못함 pd로 해야됨


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


#데이터 불러오기
# train = pd.read_csv('data/train.csv')   #EDA에서는 이것만!
# test = pd.read_csv('data/test.csv')
df = pd.read_csv('data/train.csv')


#데이터파악
print("========== head ==========")
print(df.head())
print("========== info ==========")
print(df.info())
print("========== isna ==========")
print(df.isna().sum())



#결측치 처리
df["CryoSleep"] = df["CryoSleep"].fillna(df["CryoSleep"].mode()[0])
df["Age"] = df["Age"].fillna(df["Age"].median())
df["VIP"] = df["VIP"].fillna(df["VIP"].mode()[0])
df["Cabin"] = df["Cabin"].fillna("Unknown/0/Unknown")
# df["Deck"] = df["Cabin"].str[0]
# df["HomePlanet"] = (df["HomePlanet"] == "Europa")



#단변량 분석
print("========== 단변량 분석 ==========")
print(df["Transported"].value_counts(normalize=True) * 100)

# ##범주형 시각화
# fig, axes = plt.subplots(2, 2, figsize=(20, 10))
# axes[0,0].set_title("전송됐는지 여부", fontweight="bold")
# sns.countplot(x="Transported", data=df, ax=axes[0,0])
#
# axes[0,1].set_title("냉동수면", fontweight="bold")
# sns.countplot(x="CryoSleep", data=df, ax=axes[0,1])
#
# axes[1,0].set_title("출발 행성", fontweight="bold")
# sns.countplot(x="HomePlanet", data=df, ax=axes[1,0])
#
# axes[1,1].set_title("VIP 여부", fontweight="bold")    #Cabin 이상하게 나와서 vip로 변경
# sns.countplot(x="VIP", data=df, ax=axes[1,1])
#
# ##지출액 시각화
# plt.figure(figsize=(5, 5))
# sns.boxplot(data=df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]])
# plt.title("지출액")



#이변량 분석
# fig, axes = plt.subplots(2, 2, figsize=(20, 10))
#
# axes[0,0].set_title("냉동수면 여부 / 전송 비율", fontweight="bold")
# sns.barplot(x="CryoSleep", y="Transported", data=df, ax=axes[0,0])
# axes[0,1].set_title("출발 행성 / 전송 비율", fontweight="bold")
# sns.barplot(x="HomePlanet", y="Transported", data=df, ax=axes[0,1])
#
# plt.tight_layout()
# plt.show()
#

# ##feature engineering
# ###Cabin에서 Deck 추출
# df["Deck"] = df["Cabin"].str[0]
# print(df.groupby("Deck")["Transported"].mean())
#
# axes[1,0].set_title("객실 데크 / 전송 비율", fontweight="bold")
# sns.barplot(x="Deck", y="Transported", data=df, ax=axes[1,0], order=sorted(df["Deck"].unique()))
#
#
# ###TotalSpend 구간화해서 SpendBand 만들기
# spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
# df["TotalSpand"] = df[spend_cols].sum(axis=1)
#
# df["SpendBand"] = pd.cut(
#     df["TotalSpand"],
#     bins=[0, 500, 5000, 10000, 100000],
#     labels=["지출없음", "적음", "보통", "많음"]
# )
# print(df.groupby("SpendBand")["Transported"].mean())
#
# axes[1,1].set_title("지출액 구간 / 전송 비율", fontweight="bold")
# sns.barplot(x="SpendBand", y="Transported", data= df, ax=axes[1,1])



#상관관계
# spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
# df["TotalSpand"] = df[spend_cols].sum(axis=1)
# df["Transported"] = df["Transported"].astype(int)
# df["CryoSleep"] = df["CryoSleep"].astype(int)
# df["VIP"] = df["VIP"].astype(int)
#
# cols = ["Transported", "CryoSleep", "VIP", "TotalSpand"]
# cor = df[cols].corr()
#
# plt.figure(figsize=(10,8))
# sns.heatmap(cor, cmap="coolwarm", annot=True, fmt=".2f")


# plt.tight_layout()
# plt.show()



#인사이트
'''
참고사항:
타이타닉에서 생존 (survived) 여부가 중요했다면 우주선 타이타닉에서는 전송이 (transported) 중요합니다.

냉동수면 (CryoSleep) 을 한 승객의 전송 비율이 압도적으로 높습니다. (대략 0.8 / 0.3)
또한 지출이 없는 승객일수록 전송 비율이 높아,
냉동수면 상태에서는 지출 자체가 불가능하므로,
지출 없음이 냉두수면일 가능성이 높고, 두 변수가 서로 얽혀있는 것으로 보입니다.
Europa 출신 승객의 전송 비율이 가장 높고 지구 (Earth) 출신이 가장 낮습니다.
B, C 데크(Deck) 승객의 전송 비율이 높은 편이며,
냉동수면 승객이나 특정 출발 행성 (HomePlanet) 승객이 많이 몰려 있어서,
그 효과가 반영된 것일 가능성이 큽니다.
'''

