import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
X = np.array([
    [1, 2],
    [2, 3],
    [4, 5],
    [3, 6],
    [5, 8]
])
y = np.array([5, 7, 11, 10, 14])

model=LinearRegression()
model.fit(X,y)
y_predict=model.predict(X)
#for 6.1 plot or 6.2 not plot
# plt.scatter(X,y,color='blue')
# plt.plot(X,y_predict,color='red')
# plt.title("slr")
# plt.xlabel("hour studied")
# plt.ylabel("exam")
# plt.show()
print(f"{model.coef_}")
print(f"{model.intercept_}")
print(f"{model.score(X,y)}")
print(mean_squared_error(y,y_predict))