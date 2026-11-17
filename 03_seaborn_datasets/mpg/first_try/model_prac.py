#mpg

import seaborn as sns
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#데이터
df = sns.load_dataset("mpg")


#전처리
##결측치
df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())

##features
features = ["cylinders", "displacement", "horsepower", "weight", "model_year"]

##인코딩 <origin>
result = pd.get_dummies(df["origin"], drop_first=True)
###concat
X = pd.concat([df[features], result], axis=1)
y = df["mpg"]

## train / test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#학습
m = RandomForestRegressor(random_state=42, n_estimators=100)
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


#특성 중요도
temp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
print(temp)


