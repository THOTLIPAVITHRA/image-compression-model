import numpy as np
import matplotlib.pyplot as plt

def find_closest_centroid(X, centroids):
    m, n = X.shape
    K = centroids.shape[0]
    idx = np.zeros(m, dtype = "int")
    for i in range (m):
        distances = []
        for j in range(K):
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            distances.append(norm_ij)
        idx[i] = np.argmin(distances)
    return idx

def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        points = X[idx==k]
        if points.size > 0:
            centroids[k] = np.mean(points, axis = 0)
    return centroids

def run_kMeans(X, initial_centroids, max_iters):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    for _ in range(max_iters):
        idx = find_closest_centroid(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return idx, centroids

def random_kMeans(X, K):
    m, n = X.shape
    rand_idx = np.random.permutation(m)
    centroids = X[rand_idx[:K]]
    return centroids

original_img = plt.imread("C:\\Users\\sprab\\OneDrive\\Desktop\\22001A0533\\bird.jpg")
original_img = original_img / 255  # Division for images .jpeg/.jpg format so that all values stay in range [0,1]
# This is done to ensure even scaling between different pixel values
# No  division necessary for .png files as the data is already present in plt.imread() methods
size = original_img.shape

X_img = np.reshape(original_img, (size[0]*size[1], 3))

K = 8
max_iters = 10

initial_centroids = random_kMeans(X_img, K)
idx, centroids = run_kMeans(X_img, initial_centroids, max_iters)

X_recovered = centroids[idx, :]
X_recovered = np.reshape(X_recovered, size)

fig, ax = plt.subplots(1, 2, figsize = (16, 16))

ax[0].imshow(original_img)
ax[0].set_title("Original image")
ax[0].set_axis_off()

ax[1].imshow(X_recovered)
ax[1].set_title("Compressed image, 32 colors")
ax[1].set_axis_off()

plt.show()
