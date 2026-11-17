#flights
##R2:  0.1521470918308504
##R2:  0.5580482277184193 Linear

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#데이터
df = sns.load_dataset("flights")
x = df["year"].astype(str) + "-" + df["month"].astype(str)
df["date"] = pd.to_datetime(x, format="%Y-%b")
df = df.sort_values("date").reset_index(drop=True)


#전처리
## t (time) 추세
df["t"] = np.arange(len(df))

## m (month) 계절
month_dummies = pd.get_dummies(df["month"], drop_first=True)

## train - test
X = pd.concat([df["t"], month_dummies], axis=1)
y = df["passengers"]

test_cnt = 12

X_train, X_test = X.iloc[0:-test_cnt], X.iloc[-test_cnt:]
y_train, y_test = y.iloc[0:-test_cnt], y.iloc[-test_cnt:]


#학습
# m = RandomForestRegressor(random_state=42)
m = LinearRegression()
m.fit(X_train, y_train)

#예측
y_pred = m.predict(X_test)

#오차확인
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE: ", mae)
print("RMSE: ", rmse)
print("R2: ", r2)

