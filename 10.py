# data_visualization.py

# 1. Create meaningful visualizations using data visualization tools
import seaborn as sns
import matplotlib.pyplot as plt

# Load a sample dataset (e.g., Titanic dataset)
import seaborn as sns
data = sns.load_dataset('titanic')

# 1.1 Visualize distribution of age
plt.figure(figsize=(8, 6))
sns.histplot(data['age'], kde=True)
plt.title('Age Distribution of Titanic Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 1.2 Visualize survival rate based on class
plt.figure(figsize=(8, 6))
sns.countplot(x='class', hue='survived', data=data)
plt.title('Survival Rate Based on Class')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# 2. Combine multiple visualizations to tell a compelling data story
# Let's combine a few visualizations in one figure
plt.figure(figsize=(12, 8))

# Distribution of age
plt.subplot(2, 2, 1)
sns.histplot(data['age'], kde=True)
plt.title('Age Distribution')

# Survival by Class
plt.subplot(2, 2, 2)
sns.countplot(x='class', hue='survived', data=data)
plt.title('Survival Rate Based on Class')

# Relationship between age and fare
plt.subplot(2, 2, 3)
sns.scatterplot(x='age', y='fare', hue='survived', data=data)
plt.title('Age vs Fare')

# Survival by Gender
plt.subplot(2, 2, 4)
sns.countplot(x='sex', hue='survived', data=data)
plt.title('Survival Rate by Gender')

plt.tight_layout()
plt.show()

# 3. Present the findings and insights in a clear and concise manner
# After visualizing, we can summarize findings:
print("Insights from the Titanic Dataset:")
print("- Age distribution shows a relatively normal distribution.")
print("- Passengers in first class had a higher survival rate.")
print("- There seems to be a positive correlation between age and fare.")
print("- Females had a higher survival rate than males.")
