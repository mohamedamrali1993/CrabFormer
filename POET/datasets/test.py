import random
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

class RandomBrightness(object):
    def __init__(self, brightness_range):
        self.brightness_range = brightness_range

    def __call__(self, img, target=None):
        brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        brightness_transform = T.ColorJitter(brightness=brightness_factor)
        img = brightness_transform(img)
        if target is not None:
            return img, target
        else:
            return img
        
class RandomContrast(object):
    def __init__(self, contrast_range):
        self.contrast_range = contrast_range

    def __call__(self, img, target=None):
        contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
        contrast_transform = T.ColorJitter(contrast=contrast_factor)
        img = contrast_transform(img)
        if target is not None:
            return img, target
        else:
            return img

class RandomSaturation(object):
    def __init__(self, saturation_range):
        self.saturation_range = saturation_range

    def __call__(self, img, target=None):
        saturation_factor = random.uniform(self.saturation_range[0], self.saturation_range[1])
        saturation_transform = T.ColorJitter(saturation=saturation_factor)
        img = saturation_transform(img)
        if target is not None:
            return img, target
        else:
            return img

transform = T.Compose([
    T.ToTensor(),
    #RandomBrightness(brightness_range=(0.3, 0.3)),
    #RandomContrast(contrast_range=(0.5, 0.5)),
    RandomSaturation(saturation_range=(0.3, 0.3))
])

# Load an image
img = Image.open('/home/bmv/detectron2/Crab_dataset/Easy_case/train/image1.png')

# Apply the transformations
transformed_img = transform(img)

plt.imshow(transformed_img.permute(1, 2, 0))  
plt.show()