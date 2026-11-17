#tips
##0.6208580000398983
##0.6639235737596481 사이즈 추가

import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#데이터 준비
df = sns.load_dataset("tips")


#학습용
X = df[["total_bill"]]
y = df["tip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#학습 (수치형)
m = LinearRegression()
m.fit(X_train, y_train)


#예측
y_pred = m.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(mae)

