#breast_cancer
##ACC_S:  0.9736842105263158

from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC

#지도학습

#--------- 로지스틱 회귀 <분류형> ----------
# X,y = load_breast_cancer(return_X_y=True)  #처음부터 X,y 로 데이터 받아오기
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X,y, test_size=0.2, random_state=42
# )
#
# pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
#
# #학습
# pipeline = pipeline.fit(X_train, y_train)
#
# #예측
# y_pred = pipeline.predict(X_test)
#
# #평가
# acc_s = accuracy_score(y_test,y_pred)
#
# print("ACC_S: ", acc_s)
#
# proba = pipeline.predict_proba(X_test)
#
# print("PROBA: ", np.round(proba, 3))





# ========== KNN ==========
#wine
##ACC_S:  0.9444444444444444

# data = load_wine()
#
# print(data.target_names)
#
# X,y = load_wine(return_X_y=True)
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X,y , test_size=0.2, random_state=42
# )
#
# pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
#
# pipeline = pipeline.fit(X_train,y_train)
#
# y_pred = pipeline.predict(X_test)
#
# acc_s = accuracy_score(y_test,y_pred)
#
# print("ACC_S: ", acc_s)
#
# proba = pipeline.predict_proba(X_test)
#
# print("PROBA: ", np.round(proba, 3))





# ========== SVM ==========
#wine
##ACC_S:  0.9722222222222222   <kernel=linear>
##ACC_S:  1.0   <kernel=rbf (기본값임)>


# X, y = load_wine(return_X_y=True)
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X,y, test_size=0.2, random_state=42
# )
#
# pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
#
# pipeline = pipeline.fit(X_train,y_train)
#
# y_pred = pipeline.predict(X_test)
#
# acc_s = accuracy_score(y_test,y_pred)
#
# print("ACC_S: ", acc_s)
#
# '''
# #proba
# svc는 확률로 하는 게 아니라서
# 경계선을 두고 넘었냐 안넘었냐 보는 거라서 적용 안됨
# '''
# # proba = pipeline.predict_proba(X_test)  #probability=True proba 때문에 넣음
# #
# # print("PROBA: ", np.round(proba, 3))





# ========== 모델별 성능 비교 ==========

# X,y = load_iris(return_X_y=True)
X,y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# p1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
# p2 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
# p3 = make_pipeline(StandardScaler(), SVC())

# pipelines = [p1, p2, p3]

# for pipeline in pipelines:
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     acc_s = accuracy_score(y_test, y_pred)
#     print("ACC: ", acc_s)


models = {
    "LR(로지스틱)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "SVC": make_pipeline(StandardScaler(), SVC()),
    "RF(랜덤포레스트)": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=1001))
}

for model_name, model_pipeline in models.items():
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    acc_s = accuracy_score(y_test, y_pred)
    print(model_name + " ACC_S: ", acc_s)

