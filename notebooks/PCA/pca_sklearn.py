from sklearn.decomposition import PCA
import numpy as np

# Generate some random data
X = np.random.normal(size=(100, 4))

# Create a PCA object with two components
pca = PCA(n_components=2)

# Fit the PCA model to the data
pca.fit(X)

# Transform the data to the lower-dimensional space
X_pca = pca.transform(X)

# Print the variance explained by each component
print("Variance explained by each component: ", pca.explained_variance_ratio_)

# Plot the transformed data
import matplotlib.pyplot as plt
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.show()
