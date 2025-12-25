import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def add_gaussian_noise(image, mean_range=(0, 1), std_range=(1, 3)): # 0 to 1 and 5 to 10
    mean = np.random.uniform(*mean_range)
    std = np.random.uniform(*std_range)
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return noisy_image.clip(0, np.inf)  # Ensuring no negative values

def depth_shift(image, shift_range=(-1, 5)): # -10 to 10
    shift_value = np.random.uniform(*shift_range)
    shifted_image = image + shift_value
    return shifted_image.clip(0, np.inf)

def elastic_distortion(image, alpha_range=(10, 25), sigma_range=(2, 3)): # 50 to 75 and 2 to 3
    alpha = np.random.uniform(*alpha_range)
    sigma = np.random.uniform(*sigma_range)
    random_state = np.random.RandomState(None)
    shape = image.shape
    dx = cv2.GaussianBlur((random_state.rand(*shape[:2]) * 2 - 1).astype(np.float32), (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape[:2]) * 2 - 1).astype(np.float32), (17, 17), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    distorted_image = np.zeros_like(image)

    distorted_image = cv2.remap(image.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return distorted_image

def apply_gaussian_blur(image, kernel_size_range=(5, 15)): # 15 to 75
    min_kernel_size, max_kernel_size = kernel_size_range
    kernel_size = np.random.randint(min_kernel_size, max_kernel_size)  # Ensure max is exclusive
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    blurred_image = cv2.GaussianBlur(image.astype(np.float32), (kernel_size, kernel_size), 0)
    return blurred_image

def cutout(image, mask_size_range=(10, 50)): # 50 to 200
    min_mask_size, max_mask_size = mask_size_range
    mask_size = np.random.randint(min_mask_size, max_mask_size + 1)  # Ensure max is exclusive
    mask_x = np.random.randint(0, image.shape[1] - mask_size)
    mask_y = np.random.randint(0, image.shape[0] - mask_size)
    cutout_image = image.copy()
    cutout_image[mask_y:mask_y + mask_size, mask_x:mask_x + mask_size] = 0
    return cutout_image

def dropout(image, dropout_ratio_range=(0.02, 0.05)): # 0.02 to 0.2
    dropout_ratio = np.random.uniform(*dropout_ratio_range)
    mask = np.random.binomial(1, 1 - dropout_ratio, image.shape)
    dropout_image = image * mask
    return dropout_image

def jitter(image, jitter_range=(-1, 5)): # -10 to 10
    jitter_min, jitter_max = jitter_range
    jitter = np.random.uniform(jitter_min, jitter_max, image.shape)
    jittered_image = image + jitter
    return jittered_image.clip(0, np.inf)


