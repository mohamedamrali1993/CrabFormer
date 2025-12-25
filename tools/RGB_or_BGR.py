"""This script visualizes a given numpy array as both RGB and BGR formats
to help determine its correct color channel arrangement."""

import numpy as np
import matplotlib.pyplot as plt

def visualize_rgbd(array):
    """
    Visualize the numpy array as both RGB and BGR to determine its format.
    
    Args:
    array (numpy.ndarray): The input array with shape (H, W, 4).
    """
    # Check if the input array has the correct shape
    if array.shape[-1] != 4:
        raise ValueError("The input array must have 4 channels (H, W, 4).")

    # Extract RGB and BGR representations
    rgb_array = array[..., :3]
    bgr_array = array[..., [2, 1, 0]]

    # Display the RGB image
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(rgb_array/255)
    plt.title('RGB Representation')

    # Display the BGR image
    plt.subplot(1, 2, 2)
    plt.imshow(bgr_array/255)
    plt.title('BGR Representation')

    plt.show()

# Example usage
# Create a dummy RGBD array
rgbd_array = np.load("./Datasets/image1.npy")
rgbd_array = rgbd_array.transpose(1, 2, 0)

visualize_rgbd(rgbd_array)
