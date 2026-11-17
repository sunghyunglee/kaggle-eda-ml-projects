#mpg
## R2:  0.8437364518393105 (LinearRegression)
## R2:  0.9170005276292799 (RandomForestRegressor)

import numpy as np
import seaborn as sns
import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#데이터 준비
df = sns.load_dataset("mpg")


#전처리
##결측치
df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
### 실무에서 목표가 최고 성능
# features = ['cylinders', 'displacement', 'horsepower', 'weight', 'model_year']
### 실무 목표 간단하고 해석 쉬운 모델
features = ['weight', 'model_year']     #m.coef 분석 후

##인코딩
result = pd.get_dummies(df["origin"], drop_first=True)
###행 단위로 합치기 (concat)
X = pd.concat([df[features], result], axis=1)
y = df["mpg"]

## train / test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#학습
m = LinearRegression()
# m = RandomForestRegressor(random_state=42, n_estimators=100)
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


# 회귀 개수
print(m.coef_)  #m.coef_ 는 LinearRegression()만 됨 RandomForestRegressor() 안됨
temp = pd.Series(m.coef_, index=X.columns)
print(temp)

