"""This script processes a COCO-format JSON file containing image annotations with RLE-encoded masks.
It decodes the RLE masks, computes the average depth value for each instance using the depth channel
from corresponding numpy image files, and identifies the top-most instance in each image based on depth.
The top-most instances are then saved into a new JSON file."""

import numpy as np
import json
import os
import pycocotools.mask as mask_util

# Paths
json_path = "/home/bmv/crab_loading_cv/labels/Piled_intact_core_val.json"
numpy_image_path = '/home/bmv/crab_loading_cv/Pile_core_numpy_files/'
save_path = "/home/bmv/crab_loading_cv/labels/Piled_intact_core_top_instance_val.json"
numpy_masks_path = '/home/bmv/crab_loading_cv/top_instance_numpy_mask/'

# Load JSON file
with open(json_path, 'r') as f:
    data = json.load(f)

# Check for required keys
required_keys = ['images', 'annotations', 'categories']
for key in required_keys:
    if key not in data:
        raise ValueError(f"JSON file is missing required key: {key}")

top_instances = {
    'images': [],
    'annotations': [],
    'categories': data['categories']
}

# Process each image
for image in data['images']:
    image_name = image['file_name']
    image_id = image['id']

    # Find all annotations for the image
    annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]

    if not annotations:
        continue

    # Translate RLE instances to numpy arrays
    depth_channel = None
    average_depth_values = []

    # Removing underscores for actual numpy image
    modified_image_name = image_name.replace('_', '').replace('.tiff', '.npy')
    numpy_file_path = os.path.join(numpy_image_path, modified_image_name)

    if not os.path.exists(numpy_file_path):
        continue

    actual_numpy_file = np.load(numpy_file_path)

    # Load the depth channel
    if depth_channel is None:
        if actual_numpy_file.shape[0] < 4:
            continue
        depth_channel = actual_numpy_file[3]  # Index 3 corresponds to the Depth channel

    for annotation in annotations:
        rle = annotation.get('segmentation')
        if not rle:
            continue

        if isinstance(rle, dict):
            # Convert the segmentation to the format expected by pycocotools
            rle = [rle]

        try:
            mask = mask_util.decode(rle)
        except Exception as e:
            continue

        # Convert mask to numpy array (boolean or 0s and 1s)
        mask_array = np.array(mask, dtype=np.uint8).squeeze()
        save_numpy_array = os.path.join(numpy_masks_path, image_name.replace('.tiff', '.npy'))
        np.save(save_numpy_array, mask_array)

        # Compute the depth heights for the instance by matrix multiplying the mask with the depth channel
        instance_depth_values = depth_channel * mask_array

        # Compute the average depth value for the mask
        if np.sum(mask_array) > 0:
            average_depth_value = np.sum(instance_depth_values) / np.sum(mask_array)
        else:
            average_depth_value = np.inf  # Assign infinity if the mask is empty

        annotation['average_depth_value'] = average_depth_value
        average_depth_values.append(average_depth_value)

    # Filter out annotations with inf depth value
    valid_annotations = [ann for ann in annotations if ann['average_depth_value'] != np.inf]

    if valid_annotations:
        # Sort annotations by average depth value
        sorted_annotations = sorted(valid_annotations, key=lambda ann: ann['average_depth_value'], reverse=True)

        # Get the top-most instance (annotation with the highest average depth value)
        #top_most_annotation = sorted_annotations[:-2]
        top_most_annotation = sorted_annotations[0]

        # Save top-most annotation to top_instances dictionary
        top_instances['images'].append(image)
        top_instances['annotations'].append(top_most_annotation)

# Save all top instances to JSON
with open(save_path, 'w') as f:
    json.dump(top_instances, f)