import pandas as pd
df=pd.read_csv("employee_data.csv")
print(df)
df.head(10)
print("##########")
dff=df.fillna(value=0)
print(dff)
print("##########")

dfff=df.dropna()
print(dfff)
print("##########")
#filter
high_slaray=df[df['Salary']>60000]
print("high",high_slaray)
print("##########")
#sorting
sorted_by_Salary=df.sort_values(by='Salary',ascending=False)
print("sorted",sorted_by_Salary)
print("##########")


#grouping.
count_by_dept=df['Department'].value_counts()
print("grouped",count_by_dept)
print("##########")

js=pd.read_json("emp.json")
print(js)
#
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Hardcoded data
data = {
    'age': [25, 32, 47, 51, 62],
    'salary': [50000, 60000, 80000, 90000, 120000]
}
df = pd.DataFrame(data)

# Standardization (Z-score scaling)
standard_scaler = StandardScaler()
standardized = standard_scaler.fit_transform(df)

# Normalization (Min-Max scaling)
min_max_scaler = MinMaxScaler()
normalized = min_max_scaler.fit_transform(df)

# Convert to DataFrame for better readability
standardized_df = pd.DataFrame(standardized, columns=df.columns)
normalized_df = pd.DataFrame(normalized, columns=df.columns)

print("Standardized Data:")
print(standardized_df)

print("\nNormalized Data:")
print(normalized_df)
#3.2
import pandas as pd

# Sample data
data = {
    'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red'],
    'Size': ['S', 'M', 'L', 'S', 'M'],
    'Price': [10.99, 12.50, 9.99, 11.50, 13.00]
}

# Create DataFrame
df = pd.DataFrame(data)

# Perform dummification
df_dummies = pd.get_dummies(df, columns=['Color', 'Size'])

# Display result
print(df_dummies)
#4.1
import numpy as np
from scipy import stats

# Sample scores
traditional_scores = [75, 80, 85, 70, 90]
new_method_scores = [88, 92, 85, 95, 90]

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(new_method_scores, traditional_scores)

print("T-Statistic:", t_stat)
print("P-Value:", p_value)

# Interpret result
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The new method has a significant effect.")
else:
    print("Fail to reject the null hypothesis: No significant difference detected.")
#4.2
import pandas as pd
from scipy.stats import chi2_contingency

# Contingency table: Gender vs. Product Preference
data = pd.DataFrame({
    'Product A': [30, 20],  # [Male, Female]
    'Product B': [10, 25],
    'Product C': [20, 30]
}, index=['Male', 'Female'])

# Perform chi-square test
chi2, p, dof, expected = chi2_contingency(data)

print("Chi-Square Statistic:", chi2)
print("P-Value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", expected)

# Interpret result
alpha = 0.05
if p < alpha:
    print("Reject the null hypothesis: There is a significant relationship.")
else:
    print("Fail to reject the null hypothesis: No significant relationship.")

