import seaborn as sns
import matplotlib.pyplot as plt
data=sns.load_dataset('titanic')

# visualize age distribution

plt.figure(figsize=(8,6))
sns.histplot(data['age'],kde=True)
plt.title("age dist ")
plt.xlabel('age')
plt.ylabel('frequency')
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='class',hue='survived',data=data)
plt.title("survival based on class")
plt.xlabel('class')
plt.ylabel('count')
plt.show()


#2
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
sns.histplot(data['age'],kde=True)
plt.title("age")

plt.subplot(2,2,2)
sns.countplot(x='class',hue='survived',data=data)
plt.title("class")

plt.subplot(2,2,3)
sns.scatterplot(x='age',y='fare',hue='survived',data=data)

plt.subplot(2,2,4)
sns.countplot(x='class',hue='survived',data=data)

plt.tight_layout()
plt.show()
print("Insights from the Titanic Dataset:")
print("- Age distribution shows a relatively normal distribution.")
print("- Passengers in first class had a higher survival rate.")
print("- There seems to be a positive correlation between age and fare.")
print("- Females had a higher survival rate than males.")
