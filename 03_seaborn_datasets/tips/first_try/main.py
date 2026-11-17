#tips
##누가 팁을 많이 줄까?

import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#data
df = sns.load_dataset("tips")
df.to_csv("data/tips.csv")

print(df.head())
print(df.tail())
print(df.info())
print(df.describe(include="category"))

#상관관계
# plt.figure(figsize = (8,6))
# sns.scatterplot(data=df, x="total_bill", y="tip", hue="smoker")
# plt.title("Total Bill - Tip 상관관계", fontweight="bold")
#
# plt.tight_layout()
# plt.show()

#숫자형(만) 상관관계
'''
-> data 양이 중요한 게 아니라 질이 중요함 그래서 size를 뺌 (유의미한 값이 중요함)
나중에 모델한데 값을 정할때는 eda에서 다 해보고 유의미한 값들만 정리해서 넣어
양보다는 질!
'''
num_cols = ['total_bill', 'tip', 'size']
x = df[num_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(x, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("total_bill / tip / size")

plt.tight_layout()
plt.show()


