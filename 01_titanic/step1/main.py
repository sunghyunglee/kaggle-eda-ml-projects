import pandas as pd

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# print(train.info())
# print(train.isna().sum())
# print(test.info())
# print(test.isna().sum())


# train["Age"] = train["Age"].fillna(train["Age"].median())
# result = train.groupby("Pclass")["Age"].mean()
# result = train.groupby("Pclass")["Age"].transform("median")
# train["Age"] = train["Age"].fillna(result)
# print(result)

#결측치 처리
train["Age"] = train["Age"].fillna(train.groupby("Pclass")["Age"].transform("median"))


#결측치 갯수 확인
print(train.isna().sum())


