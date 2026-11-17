#penguins
##ACC: 1.0 (RandomForestClassifier)
##ACC: 0.9855072463768116 (LogisticRegression)

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#data
df = sns.load_dataset("penguins")


#전처리
##결측치
cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
df[cols] = df[cols].fillna(df[cols].median())
df["sex"] = df["sex"].fillna(df["sex"].mode()[0])

##특징
features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

##인코딩
result = pd.get_dummies(df["island"], drop_first=True)

## concat
X = pd.concat([df[features], result], axis=1)
y = df["species"]

## train / test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)


#학습
m = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
# m = LogisticRegression(max_iter=5000)
m.fit(X_train, y_train)

#예측
y_pred = m.predict(X_test)

#모델평가
acc = accuracy_score(y_test, y_pred)
print("ACC:", acc)

