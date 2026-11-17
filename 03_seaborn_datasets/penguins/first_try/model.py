#penguins
##accuracy_score:  0.9701492537313433   (RandomForestClassifier)
##accuracy_score:  0.9850746268656716   (LogisticRegression)

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

df = sns.load_dataset("penguins")
df.to_csv("data/penguins.csv")

df = df.dropna().reset_index(drop=True)


#train, test
features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X = df[features]    #문제
y = df["species"]   #정답

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size = 0.2,
    stratify = y,
    random_state = 42
)


#학습
# m = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
m = LogisticRegression(max_iter=5000)
m.fit(X_train, y_train)


#예측
y_pred = m.predict(X_test)
# print(X_test)
# print(y_pred)


#결과
acc = accuracy_score(y_test, y_pred)
print("accuracy_score: ", acc)


#혼동행렬
lbs = ["Gentoo", "Chinstrap", "Adelie"]
cm = confusion_matrix(y_test, y_pred, labels= lbs)
# print("cm: \n", cm)
fix, ax = plt.subplots(figsize = (8,6))
sns.heatmap(data=cm, annot=True, fmt="d", ax=ax, xticklabels=lbs, yticklabels=lbs, cmap="Blues")
plt.xlabel("예측")
plt.ylabel("실제")
ax.set_title("종 혼동행렬", fontweight="bold")

result = classification_report(y_test, y_pred, labels=lbs)
print(result)


plt.tight_layout()
plt.show()


