import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

data = pd.read_csv("data.csv")

print(data.describe())

sns.pairplot(data)
plt.suptittle("house data relationships", y = 1.02)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.tittle("correalation heatmap")
plt.show()