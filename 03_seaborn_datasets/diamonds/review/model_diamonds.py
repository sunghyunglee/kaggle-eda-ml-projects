#diamonds
##R2_S: 0.8413260555788385 (LinearRegression)
##R2_S: 0.8763250394044277 (RandomForestRegressor)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#데이터
df = sns.load_dataset("diamonds")
df.to_csv("diamonds.csv")


#전처리
##결측치 <0으로 측정된 x,y,z 값을 처리 (결측치는 아니지만)>
df = df[(df["x"]>0) & (df["y"]>0) & (df["z"]>0)]

##인코딩


##모델
features = ["carat", "x", "y", "z"]
X = df[features]
y = df["price"]

## train - test 분리
X_train, X_test, y_train,y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#학습
m = LinearRegression()
# m = RandomForestRegressor(random_state=42, n_estimators=1000)
m.fit(X_train, y_train)

#예측
y_pred = m.predict(X_test)

#평가
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_s = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2_S:", r2_s)


#회귀 개수
sr = pd.Series(m.coef_, index=X.columns).sort_values(ascending=False)
print(sr)

#특성 중요도
# sr = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
# print(sr)

