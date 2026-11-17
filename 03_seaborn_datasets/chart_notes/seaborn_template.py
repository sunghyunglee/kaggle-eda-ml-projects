import seaborn as sns
from matplotlib import pyplot as plt

'''
df = sns.load_dataset("내장데이터이름")

fig, axes = plt.subplots(2,2, figsize=(10,10)))

sns.histplot(data=df, x="")
sns.boxplot()
sns.scatterplot()
sns.countplot()
sns.heatmap()
sns.pairplot()

plt.tight_layout()
plt.show()
'''


df = sns.load_dataset("tips")
sns.pairplot(df)
plt.show()


