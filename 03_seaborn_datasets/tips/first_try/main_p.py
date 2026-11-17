#tip

import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#data
df = sns.load_dataset("tips")
df.to_csv("data/tips2.csv")

print(df.head())
print(df.tail())
print(df.info())
print(df.describe(include = "category"))


#상관관계 <숫자형만>
## total_bill / tip
# plt.figure(figsize=(8,6))
# sns.scatterplot(data=df, x="total_bill", y="tip", hue="sex")
# plt.title("total_bill / tip 상관관계", fontweight="bold")
#
# plt.tight_layout()
# plt.show()

##total_bill / tips / size
df_nums = df[["total_bill", "tip", "size"]]
cor = df_nums.corr()
plt.figure(figsize=(8,6))
sns.heatmap(cor, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("total_bill / tips / size 상관관계", fontweight="bold")

plt.tight_layout()
plt.show()
