#california_housing
'''
R2:  0.5757877060324508
MAE:  0.5332001304956565
RMSE:  0.7455813830127764
'''

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

data = fetch_california_housing()
X = data.data
y = data.target

X = pd.DataFrame(X, columns=data.feature_names)
y = pd.DataFrame(y, columns=["target"])

#DataFrame으로 만들들어서 csv 파일로 저장
df = X.copy()
df['target'] = y
df.to_csv("california_housing_data.csv", index=False)

df.describe().to_csv("california_housing_data_desc.csv", index=True)

#가격 분포 히스토그램
# df.hist(figsize=(12,10), grid=False, bins=25)
# plt.tight_layout()
# plt.show()

#상관관계 히트맵
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# plt.tight_layout()
# plt.show()


#train - test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


pipeline = make_pipeline(StandardScaler(), LinearRegression())
pipeline.fit(X_train, y_train)


# scaler = StandardScaler()
# # scaler.fit(X_train)
# # scaler.transform(X_train)
# # scaler.transform(X_test)
# X_train_s = scaler.fit_transform(X_train)
# X_test_s = scaler.transform(X_test)
#
# m = LinearRegression()
# m.fit(X_train_s, y_train)
# y_pred = m.predict(X_test_s)
y_pred = pipeline.predict(X_test)


r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R2: ", r2)
print("MAE: ", mae)
print("RMSE: ", rmse)


