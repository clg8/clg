from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Sample data (2 features)
X = np.array([
    [1, 2],
    [2, 3],
    [4, 5],
    [3, 6],
    [5, 8]
])
y = np.array([5, 7, 11, 10, 14])

# Model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Mean Squared Error
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)

# R-squared
r_squared = model.score(X, y)
print("R-squared:", r_squared)
