#taxis  <분류형, 회귀형>
## 택시 어려움!
'''
#분류형
LR ACC_S:  0.9408983451536643
KNN ACC_S:  0.9385342789598109
SVM ACC_S:  0.9353821907013397
RF ACC_S:  0.9456264775413712

#
LR MAE:  0.7460480386931928
LR RMSE:  1.0130835333597965
LR R2_s:  0.9893944883594229
KNN MAE:  0.7384665090622538
KNN RMSE:  1.3810154861729993
KNN R2_s:  0.9802921971353603
SVR MAE:  0.7697618307796317
SVR RMSE:  1.9235098659359784
SVR R2_s:  0.9617677419693826
RF MAE:  0.6002239190412949
RF RMSE:  0.9539664961473843
RF R2_s:  0.9905961139762289
'''

import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

df = sns.load_dataset("taxis")
df = df.dropna()

print(df.info())

# 회귀로 하고 싶으면 fare / 분류로 하고 싶으면 payment
##분류 payment
# y = df["payment"]
# X = df.drop(columns=["payment", "pickup", "dropoff"])  # datetime 컬럼 직접 제외
# X = X.select_dtypes(include='number')                   # 나머지 문자형도 제외

##회귀 fare
y = df["fare"]
X = df.drop(columns=["fare", "pickup", "dropoff"])
X = X.select_dtypes(include='number')


X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=2
)


#분류형
# models = {
#     "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
#     "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
#     "SVM": make_pipeline(StandardScaler(), SVC()),
#     "RF": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=101)),
# }
#
# for model_name, model_pipeline in models.items():
#     model_pipeline.fit(X_train, y_train)
#     y_pred = model_pipeline.predict(X_test)
#     acc_s = accuracy_score(y_test, y_pred)
#     print(model_name + " ACC_S: ", acc_s)


#회귀형
models = {
    "LR": make_pipeline(StandardScaler(), LinearRegression()),
    "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor()),
    "SVR": make_pipeline(StandardScaler(), SVR()),
    "RF": make_pipeline(StandardScaler(), RandomForestRegressor())
}

for model_name, model_pipeline in models.items():
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_s = r2_score(y_test, y_pred)
    print(model_name + " MAE: ", mae)
    print(model_name + " RMSE: ", rmse)
    print(model_name + " R2_s: ", r2_s)

