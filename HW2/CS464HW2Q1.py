import os
from os import path
from PIL import Image
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from matplotlib.image import imread

# get the current working directory
ROOT_DIR = path.abspath(os.curdir)
DATA_DIR = path.join(ROOT_DIR, 'q1_dog')

def load_images(data_dir):
    images = []
    filenames = os.listdir(data_dir)

    for filename in filenames:
        filepath = os.path.join(data_dir, filename)
        img = Image.open(filepath).convert('L')  # Load as grayscale
        img_array = np.array(img)
        images.append(img_array)

    return images

images = load_images(DATA_DIR)

X = np.zeros((len(images), 3600))  # Initialize X matrix
for i, img_array in enumerate(images):
    X[i, :] = img_array.flatten()
print('Dataset has been loaded successfully!')

mean_image = np.mean(X, axis=0, keepdims=True)
X_centered = X - mean_image

plt.figure(figsize=(4, 4))
plt.axis('off')
plt.title('Mean Image of the Training Data')
data = mean_image.reshape((60, 60))
plt.imshow(data, interpolation='nearest')
plt.show()

X_cov = np.cov(X_centered, rowvar = False)
eigenvalues, eigenvectors = eig(X_cov)

# Sort by descending eigenvalues
idx = np.argsort(eigenvalues)[::-1][:10]
principal_components = eigenvectors.T

total_variance = eigenvalues.sum()
explained_vars = eigenvalues / total_variance
explained_vars = explained_vars.real
print("PVE for each principal component:\n", explained_vars[:10])
print("Total Variance explained by top 10 principal components: ",
      f"{explained_vars[:10].sum().real:.6f}")

plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, 11), explained_vars[:10], marker='o')
plt.title('Scree Plot (Top 10)')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.xticks(np.arange(1, 11))
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
principal_components = principal_components.real

for i, ax in enumerate(axes.flat):
    if i < 10:
        pc_image = principal_components[i].reshape(60, 60)  # Reshape to original image dimensions
        ax.imshow(pc_image, cmap='gray')
        ax.set_title(f"Principal Component {i+1}")
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()

# Q1/3)
def transform_image(X: np.ndarray, k: int):
    components = eigenvectors[:, 0: k]
    return (X - mean_image).reshape((1, 3600)) @ components

def reconstruct_image(X: np.ndarray, k: int):
    components = eigenvectors[:, 0: k]
    return X @ components.T + mean_image

def transform_and_reconstruct_image(image_path, k_values):
    image = Image.open(image_path).resize((60, 60))
    image_array = np.asarray(image, dtype=np.float64).reshape(3600, 1)

    # Reconstruction with different 'k' values
    for k in k_values:
        transformed = transform_image(image_array[:, 0], k)
        reconstructed = reconstruct_image(transformed, k)
        reconstructed_image = np.reshape(reconstructed, (60, 60))
        print("Total Variance explained by top",k ,"principal components: ",
              f"{explained_vars[:k].sum().real:.6f}")

        # Display the reconstructed image
        plt.figure(figsize=(5, 5))
        plt.imshow(reconstructed_image.real, cmap='gray')
        plt.title(f'Reconstruction with k = {k}')
        plt.show()

first_image_path = path.join(DATA_DIR, 'dog_0.jpeg')
image_data = imread(first_image_path)
plt.figure(figsize=(5, 5))
plt.imshow(image_data, cmap='gray')
plt.title(f'Original Image')
plt.show()
k_values = [1, 50, 250, 500, 3600]
transform_and_reconstruct_image(first_image_path, k_values)