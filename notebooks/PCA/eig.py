import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
x = np.random.normal(size=100)
y = 5 * x + np.random.normal(size=100)

# Calculate variance and covariance
var_x = np.var(x)
var_y = np.var(y)
cov_xy = np.cov(x, y)

# Calculate eigenvalues and eigenvectors of covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov_xy)

# Get the index of the maximum eigenvalue
max_eig_idx = np.argmax(eig_vals)

# Get the corresponding eigenvector and normalize it
max_eig_vec = eig_vecs[:, max_eig_idx]
max_eig_vec /= np.linalg.norm(max_eig_vec)

# Rotate the eigenvector by 90 degrees to get the second eigenvector
min_eig_vec = np.array([-max_eig_vec[1], max_eig_vec[0]])

# Print the results
print("Variance of x: ", var_x)
print("Variance of y: ", var_y)
print("Covariance of x and y: \n", cov_xy)
print("Eigenvalues: ", eig_vals)
print("Eigenvectors: \n", eig_vecs)

# Visualize the data, covariance, and eigenvectors
plt.scatter(x, y)
plt.arrow(np.mean(x), np.mean(y), np.sqrt(var_x) * max_eig_vec[0], np.sqrt(var_y) * max_eig_vec[1], head_width=0.1, head_length=0.1, color='r')
plt.arrow(np.mean(x), np.mean(y), np.sqrt(var_x) * min_eig_vec[0], np.sqrt(var_y) * min_eig_vec[1], head_width=0.1, head_length=0.1, color='r')
plt.show()
