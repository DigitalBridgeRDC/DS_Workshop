from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

cancer = load_breast_cancer() 
X_cancer, y_cancer = load_breast_cancer(return_X_y=True)

# Print the original dataset
print("Original dataset:")
df_original = pd.DataFrame(X_cancer, columns=cancer.feature_names)
print(df_original.head())

# Standardize the dataset
X_normalized = StandardScaler().fit_transform(X_cancer)

# Perform PCA with a specified number of components
n_components = 3  # choose how many principal components to keep
pca = PCA(n_components=n_components).fit(X_normalized)
X_pca = pca.transform(X_normalized) 

# Print the transformed dataset after PCA
print(f"\nTransformed dataset after PCA (keeping {n_components} components):")
df_pca = pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(1, n_components+1)])
print(df_pca.head())

# Calculate the attribute coefficients in the PCA space
attribute_coef = pca.components_.T * np.sqrt(pca.explained_variance_)

# Create a matrix of the attribute coefficients
attribute_coef_matrix = pd.DataFrame(attribute_coef, columns=[f"PC{i}" for i in range(1, n_components+1)], index=cancer.feature_names)

# Print the attribute coefficient matrix
print("\nAttribute Coefficient Matrix:")
print(attribute_coef_matrix)

# Calculate cumulative variance explained by the PCA
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Print the cumulative variance explained
print("\nCumulative variance explained by PCA:", cumulative_variance)

# Plot the original and transformed datasets
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot the original dataset
axs[0].scatter(X_cancer[:, 0], X_cancer[:, 1], c=y_cancer, cmap='viridis')
axs[0].set_xlabel(cancer.feature_names[0])
axs[0].set_ylabel(cancer.feature_names[1])
axs[0].set_title("Original Dataset")

# Plot the transformed dataset after PCA
axs[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_cancer, cmap='viridis')
axs[1].set_xlabel("PC1")
axs[1].set_ylabel("PC2")
axs[1].set_title(f"Transformed Dataset after PCA (keeping {n_components} components)")

# Plot the cumulative variance explained
fig, ax = plt.subplots()
ax.plot(range(1, len(cumulative_variance)+1), cumulative_variance, '-o')
ax.set_xlabel('Number of principal components')
ax.set_ylabel('Cumulative explained variance')
ax.axvline(x=n_components, linestyle='--', color='r', label=f"{n_components} components kept")
ax.legend()

plt.show()
