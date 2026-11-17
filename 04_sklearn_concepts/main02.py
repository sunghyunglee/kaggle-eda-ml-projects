import numpy as np
from sklearn.linear_model import LinearRegression

#지도학습 - 회귀 (선형,다항,규제)

X = np.linspace(0,100, 10)
y = X * 3 + 2
X = X.reshape(-1, 1)


m = LinearRegression()
m.fit(X, y)

#X값 구하기
print("m.coef_: ", m.coef_)
#+b 값 구하기 (+2)
print("m.intercept_: ", m.intercept_)


