from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

#데이터
# df = pd.DataFrame({
#     "age": [20,30,np.nan, 40,50],
#     "score": [100,np.nan,80,70,np.nan]
# })


#전처리
# df = df.fillna(df.mean(), inplace=True)
# df["age"] = df["age"].fillna(df["age"].mean())
# df["score"] = df["score"].fillna(df["score"].median())
# print(df)

##결측치
# imputer = SimpleImputer(strategy='median').set_output(transform="pandas")  #기본값은 mean
# result = imputer.fit_transform(df)
# print(result)

##스케일링  #KNN - 거리 기반으로 동작 / RandomForest 에서는 X
# X = np.array([[1.0, 100.0],
#               [2.0, 300.0],
#               [3.0, 500.0],
#               [4.0, 700.0]])
#
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# print(X_scaled)

##범주형 인코딩
'''
순위 x - get_dummies()
순위 o - .mpa()
'''
###get_dummies
# df = pd.DataFrame({"color": ["red", "green", "blue", "green"]})
# # color_d = pd.get_dummies(df["color"], drop_first=True)
# # print(color_d)
# result = pd.get_dummies(df, columns=["color"], drop_first=True)
# print(result)

###OneHotEncoder    <sklearn>
# enc = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
# encoded_df = enc.fit_transform(df[["color"]])
# df = pd.concat([df.drop(columns=["color"]), encoded_df], axis=1)
# print(df)

###map  / OrdinalEncoder
# df = pd.DataFrame({
#     "size": ["소", "대", "중", "소"],
#     "grade" : ["Gold", "Bronze", "Silver", "Bronze"]
# })
# # df["size"] = df["size"].map({"소": 0, "중": 1, "대": 2})

# enc = OrdinalEncoder(categories=[
#     ["소", "중", "대"],
#     ["Bronze", "Silver", "Gold"]
# ])
# df[["size", "grade"]] = enc.fit_transform(df[["size", "grade"]])
# print(df)

##파생칼럼, 중복제거, 시계열





#iris
##ACC_S:  1.0

#데이터   <iris>
X,y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)

#전처리
scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

#모델
m = KNeighborsClassifier(n_neighbors=3) #classifier니까 분류

#파이프라인  <전처리(스케일링)+모델>
pipe = make_pipeline(scaler, m)


#학습
# m.fit(X_train, y_train)
pipe.fit(X_train, y_train)

#예측
# y_pred = m.predict(X_test)
y_pred = pipe.predict(X_test)

#평가
acc_s = accuracy_score(y_test, y_pred)

print("ACC_S: ", acc_s)
print(y_pred)

