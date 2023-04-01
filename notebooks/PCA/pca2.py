from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.datasets import load_breast_cancer
import pandas as pd

cancer = load_breast_cancer() 
X_cancer, y_cancer = load_breast_cancer(return_X_y=True)

# Print the original dataset
print("Original dataset:")
df_original = pd.DataFrame(X_cancer, columns=cancer.feature_names)
print(df_original.head())

# Standardize the dataset
X_normalized = StandardScaler().fit_transform(X_cancer)

# Perform PCA
pca = PCA(n_components=2).fit(X_normalized)
X_pca = pca.transform(X_normalized) 

# Print the transformed dataset after PCA
print("\nTransformed dataset after PCA:")
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
print(df_pca.head())

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
axs[1].set_title("Transformed Dataset after PCA")

plt.show()
