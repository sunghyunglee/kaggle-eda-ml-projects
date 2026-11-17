#flights
##R2: 0.5580482277184156 (LinearRegression)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#데이터
df = sns.load_dataset("flights")
df.to_csv("flights.csv")

print(df.head())
print(df.tail())
print(df.info())
print(df.isna().sum())
print(df.describe(include='category'))
print(df["month"].value_counts())


#전처리
df["date"] = pd.to_datetime(
    df["year"].astype(str) + "-" + df["month"].astype(str),
    format="%Y-%b",
)

df = df.sort_values("date").reset_index(drop=True)

##인코딩
month_d = pd.get_dummies(df["month"], drop_first=True)

###concat
# df_c = pd.concat([df[["passengers"]], month_d], axis=1)

month_d["t"] = np.arange(len(df))


#모델   <시계열>
X = month_d
y = df["passengers"]

n_test = 12

#train - test   <시계열>
# X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
# y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
X_train, y_train = X.iloc[:-n_test], y.iloc[:-n_test]
X_test, y_test = X.iloc[-n_test:], y.iloc[-n_test:]


#학습
m = LinearRegression()
m.fit(X_train, y_train)

#예측
y_pred = m.predict(X_test)

#평가
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_s = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2_s)



