import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import io, color

# Load image
image = io.imread('cat.jpg')
# Convert image to grayscale
gray_image = color.rgb2gray(image)

# Sobel edge detection
kernel_sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel_sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Gaussian blurring
kernel_gaussian = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

# Laplacian sharpening
kernel_laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

# Embossing
kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

# Define kernel
neutral = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

kernel = kernel_sobel_y

# Perform convolution
convolved_image = convolve2d(gray_image, kernel, mode='same')

# Plot original and convolved images side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(gray_image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(convolved_image, cmap='gray')
ax2.set_title('Convolved Image')
plt.show()
