#k-fold 교차검증
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

X,y = load_breast_cancer(return_X_y=True)

models = {
    "LF": LogisticRegression(max_iter=5000),
    "RF": RandomForestClassifier(n_estimators=1001),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
}

for model_name, model in models.items():
    result = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"{model_name}")
    print(result.max())
    print(result.mean())
    print(result.std())



