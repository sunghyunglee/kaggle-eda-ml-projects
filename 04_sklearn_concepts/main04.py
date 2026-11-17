#breast_cancer   범주형

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

'''
모델평가 및 검증
교차검증 / 지표 / 과적합
'''

#===과적합===
'''
max_depth=3 : 테스트 0.9473684210526315 / 훈련 0.978021978021978점
'''

# X,y = load_breast_cancer(return_X_y=True)
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X,y, test_size = 0.2, random_state = 42
# )
#
# # m = DecisionTreeClassifier(max_depth = 6, random_state=42)
# for depth in range(1, 11):
#     m = DecisionTreeClassifier(max_depth = depth, random_state=42)  #depth가 깊어 질수록 train 올라갈 수 밖에 없음
#
#     m.fit(X_train, y_train)
#     y_pred = m.predict(X_test)
#     y_pred_train = m.predict(X_train)
#
#     acc_s_test = accuracy_score(y_test, y_pred)
#     acc_s_train = accuracy_score(y_train, y_pred_train)
#
#     # print(f"===== {depth} 회차 =====")
#     # print("ACC_S_TEST: ", acc_s_test)
#     # print("ACC_S_TRAIN: ", acc_s_train)
#     print(f"max_depth={depth} : 테스트 {acc_s_test} / 훈련 {acc_s_train}점")



# #===K-fold 교차검증===
# X,y = load_breast_cancer(return_X_y=True)
#
# # m = LogisticRegression(max_iter=5000)
#
# models = {
#     "LF": LogisticRegression(max_iter=5000),
#     "KNN": KNeighborsClassifier(n_neighbors=5),
#     "SVC": SVC(),
#     "RF": RandomForestClassifier(n_estimators=1001)
# }
#
# for model_name, model, in models.items():
#     result = cross_val_score(model, X, y, cv=5, scoring='accuracy')
#
#     print(f"{model_name}")
#     print(result.max())
#     print(result.mean())
#     print(result.std())



#===K-fold (교차검증): Stratified===
# X,y = load_breast_cancer(return_X_y=True)
#
# m = LogisticRegression(max_iter=5000)
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# results = cross_val_score(m, X, y, cv=skf, scoring='accuracy')
# print(results.mean())
# print(results.std())
# print(results.max())



#===K-fold (교차검증) : 모든 지표 한번에 ===
# X,y = load_breast_cancer(return_X_y=True)
#
# m = LogisticRegression(max_iter=5000)
# scoring = ["accuracy", "precision", "recall", "f1"]
# result = cross_validate(m, X, y, cv=5, scoring=scoring)
#
# for k in result:
#     print(f"{k} : {result[k]}")



#===혼동 행렬===
'''
정확도로만 판단하면 안됨
    -> 지금처럼 불균형 데이터는 정확도로 판단하면 안됨
'''
# X, y = make_classification(
#     n_samples=1000,
#     n_features=6,
#     n_informative=4,
#     n_classes=3,
#     weights=[0.9, 0.05, 0.05],
#     random_state=42,
# )
#
# # print(np.bincount(y))
#
# y_pred = np.zeros_like(y)
# acc_s = accuracy_score(y, y_pred)
# print("ACC_S: ", acc_s)
#
# cm = confusion_matrix(y, y_pred)
# print("CM: ", cm)



#===혼동 행렬===
X,y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

m = LogisticRegression(max_iter=5000)
m.fit(X_train, y_train)
y_pred = m.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

cr = classification_report(y_test, y_pred)
print(cr)



