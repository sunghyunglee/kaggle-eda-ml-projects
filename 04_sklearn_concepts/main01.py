#iris
##ACC_S:  1.0

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X,y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))

pipeline = pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

acc_s = accuracy_score(y_test,y_pred)

print("ACC_S: ", acc_s)


