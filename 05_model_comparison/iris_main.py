#iris   <분류형>
'''
LR ACC_S:  1.0
KNN ACC_S:  1.0
SVM ACC_S:  1.0
RF ACC_S:  1.0
'''

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X,y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)

models = {
    "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "SVM": make_pipeline(StandardScaler(), SVC()),
    "RF": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=101))
}

for model_name, model_pipeline in models.items():
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    acc_s = accuracy_score(y_test, y_pred)
    print(model_name + " ACC_S: ", acc_s)

