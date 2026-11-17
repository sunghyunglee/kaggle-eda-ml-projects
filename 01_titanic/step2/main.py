import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


#preprocess (전처리)
for df in [train, test]:
    df["Age"] = df["Age"].fillna(df.groupby("Pclass")["Age"].transform("median"))
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Sex"] = (df["Sex"] == "female").astype(int)


#feature selection
features = ["Sex", "Pclass", "Age"]
X = train[features]     #문제
y = train["Survived"]


#train
m = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3)
m.fit(X, y)     #학습


#predict
result = m.predict(test[features])   #시험지
test["Survived"] = result


#save results
test[["PassengerId", "Survived"]].to_csv('data/result.csv', index=False)


