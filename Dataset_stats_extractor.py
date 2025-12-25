""" This script computes the mean and standard deviation of a dataset of images stored as .npy files."""

import numpy as np
import os

def incremental_mean_std(file_path, mean, M2, count):
    # Load the image
    image = np.load(file_path)

    # Update the count of processed images
    count += 1

    # Compute the new mean and M2
    delta = image - mean
    mean += delta / count
    delta2 = image - mean
    M2 += delta * delta2

    return mean, M2, count

# Directory containing the dataset
dataset_dir = 'Datasets'

# Define the ranges
# start1, end1 = 1, 700
# start2, end2 = 1100, 1700
start1, end1 = 700 , 1100

# Initialize the mean, M2, and count
mean = np.zeros((4, 1030, 1086))
M2 = np.zeros((4, 1030, 1086))
count = 0

# Process the first range of images
for i in range(start1, end1 + 1):
    file_path = os.path.join(dataset_dir, f'image{i}.npy')
    if os.path.exists(file_path):
        mean, M2, count = incremental_mean_std(file_path, mean, M2, count)

# Process the second range of images
# for i in range(start2, end2 + 1):
#     file_path = os.path.join(dataset_dir, f'image{i}.npy')
#     if os.path.exists(file_path):
#         mean, M2, count = incremental_mean_std(file_path, mean, M2, count)

# Compute the final mean and standard deviation
std = np.sqrt(M2 / count)

# Compute the mean and std over all pixels
mean_values = np.mean(mean, axis=(1, 2))
std_values = np.mean(std, axis=(1, 2))

print("Mean values (RGBD):", mean_values)
print("Standard deviation values (RGBD):", std_values)
