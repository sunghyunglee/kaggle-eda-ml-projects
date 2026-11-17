#penguins
##ACC_s:  1.0 (RandomForestClassifier)
##ACC_s:  0.9855072463768116 (LogisticRegression)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#데이터
df = sns.load_dataset("penguins")
df.to_csv("penguins.csv")


#전처리
##결측치
cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
df[cols] = df[cols].fillna(df[cols].median())
df["sex"] = df["sex"].fillna(df["sex"].mode()[0])

##인코딩
island_d = pd.get_dummies(df["island"], drop_first=True)


#모델
##데이터 분리
features = cols

###concat
X = pd.concat([df[features], island_d], axis=1)
y = df["species"]

##train - test  -> stratify=y 빼니까 점수 잘 나옴
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#학습
# m = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
m = LogisticRegression(max_iter=5000)
m.fit(X_train, y_train)

#예측
y_pred = m.predict(X_test)

#평가
acc_s = accuracy_score(y_test, y_pred)
print("ACC_s: ", acc_s)


#회귀 계수
sr = pd.Series(m.coef_[0], index=X.columns).sort_values(ascending=False)
print(sr)

#특성 중요도
# sr = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
# print(sr)


#리포트
cr = classification_report(y_test, y_pred)
print(cr)


#혼돈행렬
# cm = confusion_matrix = confusion_matrix(y_test, y_pred)
# print(cm)
# plt.figure(figsize=[8, 6])
# sns.heatmap(
#     cm, annot=True, cmap="Blues",
#     xticklabels=["Adelie", "Chintrap", "Gentoo"],
#     yticklabels=["Adelie", "Chintrap", "Gentoo"],
# )
# plt.xlabel("실제")
# plt.ylabel("예측")
# plt.tight_layout()
# plt.show()

