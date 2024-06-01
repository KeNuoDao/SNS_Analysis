import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the data
data = {
    'Age Group': [17, 26, 23, 28, 30, 56, 19, 17, 45, 32, 22, 16, 26, 47, 23, 26, 51, 24, 28, 15, 19, 27, 51, 19, 27, 48, 15, 19, 19, 26, 24, 61, 19, 27, 15, 18, 25, 55, 19, 22],
    'Gender': [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
    'Education Level': [0, 2, 2, 3, 3, 4, 1, 0, 3, 2, 2, 0, 3, 4, 2, 2, 3, 2, 3, 0, 2, 3, 4, 2, 3, 4, 0, 2, 1, 3, 3, 4, 2, 3, 0, 2, 3, 4, 2, 3],
    'Income Level': [1, 2, 2, 3, 4, 2, 2, 1, 4, 3, 2, 1, 3, 2, 2, 3, 3, 2, 4, 1, 2, 3, 2, 2, 3, 3, 1, 2, 2, 3, 4, 2, 2, 3, 1, 2, 3, 2, 2, 3],
    'Relatedness Satisfaction': [6, 4, 5, 3, 4, 2, 7, 5, 3, 6, 4, 6, 5, 3, 6, 4, 3, 5, 4, 5, 6, 4, 3, 7, 5, 2, 6, 4, 5, 3, 4, 2, 6, 4, 5, 6, 4, 3, 7, 5],
    'Self-Presentation Satisfaction': [5, 4, 6, 3, 4, 2, 6, 4, 3, 5, 5, 6, 5, 2, 7, 3, 3, 5, 4, 4, 6, 5, 3, 7, 4, 2, 6, 5, 6, 4, 5, 3, 6, 5, 5, 6, 4, 3, 6, 5],
    'SNS Addiction': [6, 5, 5, 4, 4, 2, 7, 5, 3, 6, 5, 7, 5, 3, 6, 4, 3, 6, 5, 5, 6, 5, 3, 7, 5, 2, 6, 5, 6, 4, 4, 2, 6, 5, 5, 6, 5, 3, 7, 5]
}

df = pd.DataFrame(data)

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()


sns.pairplot(df)
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Relatedness Satisfaction', y='SNS Addiction', hue='Gender')
plt.title('SNS Addiction vs Relatedness Satisfaction')
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Self-Presentation Satisfaction', y='SNS Addiction', hue='Gender')
plt.title('SNS Addiction vs Self-Presentation Satisfaction')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Age Group', y='SNS Addiction', palette='Set3')
plt.title('SNS Addiction across Age Groups')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Education Level', y='SNS Addiction', palette='Set3')
plt.title('SNS Addiction across Education Levels')
plt.show()
