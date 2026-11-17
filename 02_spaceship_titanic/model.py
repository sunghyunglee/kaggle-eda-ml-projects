#Spaceship Titanic

##score 0.76455 <첫시도>
##최고점 0.80196 <n_estimators=1000, max_depth=9 일 때>
##EDA 반영 후 오히려 점수 떨어짐 최종 점수: 0.80009

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


#df 준비
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


#df 확인
print("=====train=====")
print(train.info())
print(train.isna().sum())
print("=====test=====")
print(test.info())
print(test.isna().sum())


#df 전처리
for df in [train, test]:
    # df["CryoSleep"] = df["CryoSleep"].fillna(df.groupby("Cabin")["CryoSleep"].transform("median"))    #cabin 여기 안됨
    df["CryoSleep"] = df["CryoSleep"].fillna(df["CryoSleep"].mode()[0])
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["HomePlanet"] = (df["HomePlanet"] == "Europa")
    df["VIP"] = df["VIP"].fillna(df["VIP"].mode()[0])
    df["Cabin"] = df["Cabin"].fillna("Unknown/0/Unknown")

#Cabin을 위한 Deck/Side 새로 만든 컬럼
    df["Deck"] = (df["Cabin"].str[0] == "F")    #"B/0/P" -> 첫 글자 "B" -> F인지 비교
    df["Side"] = (df["Cabin"].str[-1] == "P")   #"B/0/P" -> 마지막 글자 "P" -> P인지 비교

#
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df[spend_cols] = df[spend_cols].fillna(df[spend_cols].median())
    df["TotalSpend"] = df[spend_cols].sum(axis=1)


#df 특징 고르기
features = ["CryoSleep", "VIP", "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Deck", "Side"]
X = train[features]
y = train["Transported"]


#학습
m = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=9)
m.fit(X, y)


#예측
result = m.predict(test[features])
test["Transported"] = result


#파일로 저장
test[["PassengerId", "Transported"]].to_csv('data/result.csv', index=False)

