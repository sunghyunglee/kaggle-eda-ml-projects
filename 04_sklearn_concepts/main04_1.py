#diabetes   회귀형

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)

m = LinearRegression()

m.fit(X_train, y_train)
y_pred = m.predict(X_test)

MAE = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_s = r2_score(y_test, y_pred)

print("MAE: ", MAE)
print("RMSE: ", rmse)
print("R2: ", r2_s)

