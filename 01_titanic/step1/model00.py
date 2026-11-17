import pandas as pd


test = pd.read_csv('data/test.csv')

#성별에 따른 생존 여부   <여자인 경우>
test["Survived"] = (test["Sex"] == "female").astype(int)  #sr 동등비교 상관없음<가능>

print(test)

# test.to_csv("data/result.csv", index=False)

#성별에 따른 생존 여부   <여자인 경우>
test[["PassengerId", "Survived"]].to_csv("data/result.csv", index=False)


