#tips
##R2_S: 0.8371443140787371 (RandomForestRegressor)
##R2_S: 0.8058745445482826 (LinearRegression)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#데이터
df = sns.load_dataset("tips")

#전처리
##파생 변수
df["tip_pct"] = df["tip"] / df["total_bill"] * 100

##인코딩
df = pd.get_dummies(df, columns = ["sex", "smoker", "day", "time"])


#모델
X = df.drop(columns=["tip"])
y = df["tip"]

# train - test  (kaggle 할때는 안해도 됨)
X_train_, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#학습
# m = RandomForestRegressor(random_state=42, n_estimators=100)
m = LinearRegression()
m.fit(X_train_, y_train)

#예측
y_pred = m.predict(X_test)


#평가
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_s = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2_S:", r2_s)

#특성 중요도
# sr = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
# print(sr)

#회귀 계수
sr = pd.Series(m.coef_, index=X.columns).sort_values(ascending=False)
print(sr)

#여긴 회귀(Regressor)라서 classification_report 못씀 <분류(Classifier) 전용 평가지표임>


