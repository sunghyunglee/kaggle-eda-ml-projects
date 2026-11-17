#iris   (분류형)

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

#트리, 앙상블
#=== 결정트리 ===
# iris = load_iris()
# # X = iris.data; y = iris.target
# # y = iris.target
# X,y = iris.data,iris.target
#
# print(iris.feature_names)
#
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size = 0.2, random_state = 42
# )
#
# m = DecisionTreeClassifier(max_depth=3, random_state=42)
# m.fit(X_train, y_train)
# y_pred = m.predict(X_test)
#
# acc_s = accuracy_score(y_test,y_pred)
# print("ACC_S: ", acc_s)
# print("depth: ", m.get_depth())
#
# et = export_text(m, feature_names=iris.feature_names)
# print(et)



#=== 앙상블: 배깅 ===
#iris   (분류형)

# iris = load_iris()
# X,y = iris.data,iris.target
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size = 0.2, random_state = 42
# )
#
# m = RandomForestClassifier(n_estimators=101, random_state=42, max_depth=3)
#
# m.fit(X_train, y_train)
# y_pred = m.predict(X_test)
#
# acc_s = accuracy_score(y_test,y_pred)
# print("ACC_S: ", acc_s)
#
# cr = classification_report(y_test,y_pred)
# print("CR: ", cr)



#=== 앙상블: 부스팅 ===
#load_breast_cancer (분류형)

cancer = load_breast_cancer()

X,y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

m = GradientBoostingClassifier(
    n_estimators=101,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

m.fit(X_train, y_train)
y_pred = m.predict(X_test)

cr = classification_report(y_test,y_pred)
print("CR: ", cr)



