#diamonds
##R2:  0.9828071513639746

import numpy as np
import seaborn as sns
from pandas.core.common import random_state
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#데이터 준비
df = sns.load_dataset("diamonds")


#전처리: 결측치, 중복, 인코딩, train-test
##결측치
# print(df["x"] > 0)
# print((df["y"] > 0).sum())
# print((df["z"] > 0).sum())
df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)].reset_index(drop=True)

##인코딩
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clar_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

temp = []
for x in df["color"]:
    if x == "J":
        temp.append(0)
    elif x == "I":
        temp.append(1)
    elif x == "H":
        temp.append(2)
    elif x == "G":
        temp.append(3)
    elif x == "F":
        temp.append(4)
    elif x == "E":
        temp.append(5)
    elif x == "D":
        temp.append(6)

df["cut_o"] = df["cut"].map({v:i for i, v in enumerate(cut_order)})
df["color_o"] = pd.Series(temp)     #enumerate 안쓰고 하는 법
df["clar_o"] = df["clarity"].map({v:i for i, v in enumerate(clar_order)})

##train / test 분리
features = ["carat", "cut_o", "color_o", "clar_o", "x", "y", "z"]
X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#학습
# m = LinearRegression()
m = RandomForestRegressor(random_state=42, n_estimators=100)
m.fit(X_train, y_train)


#예측
y_pred = m.predict(X_test)
print(y_pred)


#모델평가
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2s = r2_score(y_test, y_pred)

print("MAE: ", mae)
print("RMSE: ", rmse)
print("R2: ", r2s)

##개수확인
# sr = pd.Series(m.coef_, index=features)
# print(sr)


