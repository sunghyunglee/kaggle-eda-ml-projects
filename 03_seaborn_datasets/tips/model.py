#tips
##R2:  0.24643925433611036
##R2:  0.5178244817316244 LinearRegression
'''
쓸때 없는 거 많이 넣어봤자 소용 없다ㅋㅋ
tips / total_bill
이거 2개다!
'''


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#데이터 준비
df = sns.load_dataset("tips")

#전처리: 인코딩, train / test
##인코딩
df["sex"] = df["sex"].map({"Male": 0, "Female": 1})
day_dummies = pd.get_dummies(df["day"], drop_first=True)
df = pd.concat([df, day_dummies], axis=1)

##train /test
features = ["sex", "total_bill"] + list(day_dummies.columns)
X = df[features]
y = df["tip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#학습
# m = RandomForestRegressor(random_state=42, n_estimators=100)
m = LinearRegression()
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

