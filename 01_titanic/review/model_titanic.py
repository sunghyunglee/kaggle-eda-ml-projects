#titanic
##ACC:  0.8100558659217877 (LogisticRegression)
##ACC:  0.8212290502793296 (RandomForestClassifier)

import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#데이터
# df = sns.load_dataset("titanic")
# train = pd.read_csv("train.csv")
# test = pd.read_csv("test.csv")
# df.to_csv("titanic.csv")
df = pd.read_csv("train.csv")


#전처리
##결측치
# for df in [train, test]:
df["Age"] = df["Age"].fillna(df["Age"].median())
# df["deck"] = df["deck"].fillna(df["deck"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

##인코딩
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

result = pd.get_dummies(df["Cabin"], drop_first=True)
# df["Deck"] = df["Cabin"].str[0].fillna("Unknown")
# deck_dummies = pd.get_dummies(df["Deck"], drop_first=True)

# df["Title"] = df["Name"].str.extract(r',\s*([^\.]+)\.')
# title_dummies = pd.get_dummies(df["Title"], drop_first=True)


##파생변수
df["FamilySize"] = df["Parch"] + df["SibSp"] + 1


#모델
##데이터 분리
features = ["Age", "Embarked", "Fare", "Sex", "FamilySize", "Pclass"]
# X = df[features]
# y = df["Survived"]

###행단위로 합치기 (concat)
X = pd.concat([df[features], result], axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# train_X = train[features]
# train_y = train["Survived"]
# test_X = test[features]
# test_y = test["Survived"]

##학습
m = RandomForestClassifier()
# m = LogisticRegression(max_iter=1000)
m.fit(X_train, y_train)

##예측
y_pred = m.predict(X_test)

##평가
acc = accuracy_score(y_test, y_pred)

print("ACC: ", acc)


#회귀 계수
'''
[0] 넣는 이유
pd.Series()는 1차원을 원하므로, 
2차원인 LogisticRegression은 [0]으로 첫 번째 행을 꺼내서 1차원으로 만들어줘야 함
'''
# sr = pd.Series(m.coef_[0], index=features).sort_values(ascending=False)
# sr = pd.Series(m.coef_[0], index=X.columns).sort_values(ascending=False)
# print(sr)

#특성 중요도
sr = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
print(sr)


#리포트
report = classification_report(y_test, y_pred, target_names=["사망", "생존"])
print("===== Report =====")
print(report)


#혼돈행렬
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=[8, 6])
# sns.heatmap(
#     cm, annot=True, fmt="g", cmap="Blues",
#     xticklabels=["사망예측", "생존예측"],
#     yticklabels=["사망실제", "생존실제"],
# )
# plt.show()


