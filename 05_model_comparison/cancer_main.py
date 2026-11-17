#breast_cancer  <분류형>
##ACC_S:  0.9736842105263158

import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

df_cancer = load_breast_cancer(as_frame=True).frame
# df_cancer.to_csv("data/cancer.csv", index=False)


X,y = load_breast_cancer(return_X_y=True)

#데이터 정보 확인
cancer = load_breast_cancer()
print(cancer.target_names)
print(cancer.data.shape)
##결측치 확인
print(np.isnan(cancer.data).sum())


#y 분포 히스토그램
# sns.histplot(y, bins=3)
# plt.title("y 분포 히스토그램", fontweight="bold")
#
# plt.tight_layout()
# plt.show()

#상관관계 히트맵
# plt.figure(figsize=(18,12))
# sns.heatmap(df_cancer.corr(), annot=True, cmap="coolwarm")
# plt.title("상관관계 히트맵", fontweight="bold")
#
# plt.tight_layout()
# plt.show()


#train - test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#파이프라인
pipeline = make_pipeline(StandardScaler(), LogisticRegression())


#학습
pipeline.fit(X_train, y_train)

#예측
y_pred = pipeline.predict(X_test)

#평가
acc_s = accuracy_score(y_test, y_pred)

print("ACC_S: ", acc_s)

