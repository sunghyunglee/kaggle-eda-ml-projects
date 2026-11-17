##랜덤포레스트 적용
###0.78468 최고점

import pandas as pd
#알트 + 엔터로 sklean 설치
from sklearn.ensemble import RandomForestClassifier


#data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


#전처리
for df in [train, test]:
    df["Age"] = df["Age"].fillna(df.groupby("Pclass")["Age"].transform("median"))
    df["Sex"] = (df["Sex"] == "female").astype(int)
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())


    #특징 만들기
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Title"] = df["Name"].str.extract(r',\s*([^\.]+)\.')
    df["Title"] = df["Title"].replace(["Mlle" , "Ms"], "Miss").replace("Mme", "Mrs")
    df["Title"] = df["Title"].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], "Rare")
    df["Title"] = df["Title"].map({"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5})


#특징 고르기
features = ["Pclass", "Sex", "Age", "Fare", "FamilySize", "Title"]
X = train[features]     #문제 to ai
y = train["Survived"]   #답 to ai


#학습
m = RandomForestClassifier(n_estimators=1000, random_state=100, max_depth=3)
m.fit(X, y)


#예측
result = m.predict(test[features])
test["Survived"] = result

test[["PassengerId", "Survived"]].to_csv("data/result.csv", index=False)



