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
