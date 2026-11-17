#diamonds
##R2:  0.8671008566614367 (LinearRegression)
##R2:  0.9823840032899259 (RandomForestRegressor)

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#데이터
df = sns.load_dataset("diamonds")


#전처리
##결축치
df = df[(df["x"]>0) & (df["y"]>0) & (df["z"]>0)].reset_index(drop=True)

##인코딩
cut_order = ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']
color_order = ['G', 'E', 'F', 'H', 'D', 'I', 'J']
clarity_order = ['SI1', 'VS2', 'SI2', 'VS1', 'VVS2', 'VVS1', 'IF', 'I1']

df["cut_o"] = df["cut"].map({v:i for i,v in enumerate(cut_order)})
df["color_o"] = df["color"].map({v:i for i,v in enumerate(color_order)})
df["clarity_o"] = df["clarity"].map({v:i for i,v in enumerate(clarity_order)})

result = df[["cut_o", "color_o", "clarity_o"]]

##특징
features = ["carat", "x", "y", "z"]

##concat
X = pd.concat([df[features], result], axis=1)
y = df["price"]

##train / test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size = 0.2, random_state = 42
)


#학습
# m = LinearRegression()
m = RandomForestRegressor(n_estimators = 100, random_state = 42)
m.fit(X_train, y_train)

#예측
y_pred = m.predict(X_test)

#모델평가
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2s = r2_score(y_test, y_pred)

print("MAE: ", mae)
print("RMSE: ", rmse)
print("R2: ", r2s)


#회귀 개수
# print(m.coef_)
# temp = pd.Series(m.coef_, index = X.columns)
# print(temp)

#특성 중요도
temp = pd.Series(m.feature_importances_, index = X.columns)
print(temp)

