import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns

#데이터 준비
df = sns.load_dataset('tips')


#학습용 / 시험용 데이터 분리
X = df[["total_bill"]]  #X는 무조건 2차원 -> [[]]
y = df["tip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state = 42
)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


#학습 <수치형>
m = LinearRegression()
m.fit(X_train, y_train)


#예측
y_pred = m.predict(X_test)
##수치형 데이터 예측    #accuracy_score는 범주형
# mean_absolute_error = m.score(X_test, y_test)
# print(mean_absolute_error)
mae = mean_absolute_error(y_test, y_pred)   #이것만 알아도됨
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(mae)
print(rmse)
print(r2)


