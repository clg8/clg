# pca_analysis.py

# 1. Perform PCA on a dataset to reduce dimensionality
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset (use your dataset here)
# Example: iris dataset
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Standardize the dataset
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# 2. Evaluate the explained variance and select the appropriate number of principal components
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Explained Variance:", explained_variance)
print("Cumulative Variance:", cumulative_variance)

# Plot explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Based on the cumulative variance, select number of components that explain enough variance (e.g., > 90%)
n_components = np.argmax(cumulative_variance >= 0.90) + 1
print(f"Number of components that explain > 90% variance: {n_components}")

# 3. Visualize the data in the reduced-dimensional space (2D for simplicity)
pca_2d = PCA(n_components=2)
pca_2d_result = pca_2d.fit_transform(scaled_data)

plt.figure(figsize=(8, 6))
plt.scatter(pca_2d_result[:, 0], pca_2d_result[:, 1], c=data.target, cmap='viridis')
plt.title('PCA - 2D Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Target')
plt.show()
