# image-compression-model
**Image Compression using K-Means Clustering**
This project demonstrates how to compress an image by reducing its number of colors using K-Means Clustering in Python. The algorithm groups similar colors into clusters and replaces the pixel values with their corresponding cluster centroids, significantly reducing the image size while preserving visual similarity.
**Files Included****
kmeans_image_compression.py: Python script implementing K-Means for image color compression.
bird.jpg: Original image used for compression.

**How It Works**
The image is read using matplotlib.pyplot.imread and normalized.
The image is reshaped into a 2D array where each row represents a pixel's RGB value.
K-Means clustering is applied to group pixels into K clusters.
Each pixel is assigned the RGB value of its closest centroid.
The modified image is reshaped and displayed side-by-side with the original.

**Output Example**
Original Image: The actual image of the bird.
Compressed Image: The image reconstructed using only 8 dominant colors.

**Parameters**
K = 8: Number of colors (clusters) to compress the image into.
max_iters = 10: Number of iterations for K-Means algorithm.
**Requirements**
Python 3
numpy
matplotlib

**Run the Script**
python kmeans_image_compression.py
**Notes****
You can change the K value to increase or decrease the number of colors in the compressed image.
Ensure the image path is correctly set in the script (bird.jpg should be in the same directory or update the path).

****Preview****
Original Image	                    Compressed Image
(generated on run)                     (8 colors)	
