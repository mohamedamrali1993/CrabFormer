import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# Read the JSON file
file_path = "/home/bmv/crab_loading_cv/poet-detectron/Poet_R50_q30_RGBD_v2/metrics.json"
with open(file_path, "r") as f:
    data = [json.loads(line) for line in f]

# Define lists of metrics (only APs and ARs)
# Segmentation (Segm) Metrics
metric_ap_easy_segm = ["crab_easy_val/segm/AP", "crab_easy_val/segm/AP50", "crab_easy_val/segm/AP75"]
metric_ap_orientation_easy_segm = ["crab_easy_val/segm/AP -Intact-belly_down", "crab_easy_val/segm/AP -Intact-belly_up"]
metric_ar_easy_segm = ["crab_easy_val/segm/AR1", "crab_easy_val/segm/AR10", "crab_easy_val/segm/AR100"]

metric_ap_medium_segm = ["crab_medium_val/segm/AP", "crab_medium_val/segm/AP50", "crab_medium_val/segm/AP75"]
metric_ap_orientation_medium_segm = ["crab_medium_val/segm/AP -Intact-belly_down", "crab_medium_val/segm/AP -Intact-belly_up"]
metric_ar_medium_segm = ["crab_medium_val/segm/AR1", "crab_medium_val/segm/AR10", "crab_medium_val/segm/AR100"]

metric_ap_pile_segm = ["crab_pile_val/segm/AP", "crab_pile_val/segm/AP50", "crab_pile_val/segm/AP75"]
metric_ap_orientation_pile_segm = ["crab_pile_val/segm/AP -Intact-belly_down", "crab_pile_val/segm/AP -Intact-belly_up"]
metric_ar_pile_segm = ["crab_pile_val/segm/AR1", "crab_pile_val/segm/AR10", "crab_pile_val/segm/AR100"]

metric_ap_pile_top_segm = ["crab_pile_val_top_instance/segm/AP", "crab_pile_val_top_instance/segm/AP50", "crab_pile_val_top_instance/segm/AP75"]
metric_ap_orientation_pile_top_segm = ["crab_pile_val_top_instance/segm/AP -Intact-belly_down", "crab_pile_val_top_instance/segm/AP -Intact-belly_up"]
metric_ar_pile_top_segm = ["crab_pile_val_top_instance/segm/AR1", "crab_pile_val_top_instance/segm/AR10", "crab_pile_val_top_instance/segm/AR100"]

metric_ap_pile_core_segm= ["crab_pile_val_core/segm/AP","crab_pile_val_core/segm/AP50", "crab_pile_val_core/segm/AP75" ] 
metric_ap__orientation_core_segm= ["crab_pile_val_core/segm/AP -Intact-belly_down","crab_pile_val_core/segm/AP -Intact-belly_up" ]
metric_ar_pile_core_segm = ["crab_pile_val_core/segm/AR1", "crab_pile_val_core/segm/AR10", "crab_pile_val_core/segm/AR100"]

metric_ap_pile_core_top_segm= ["crab_pile_val_core_top_instance/segm/AP","crab_pile_val_core_top_instance/segm/AP50", "crab_pile_val_core_top_instance/segm/AP75" ] 
metric_ap__orientation_core_top_segm= ["crab_pile_val_core_top_instance/segm/AP -Intact-belly_down","crab_pile_val_core_top_instance/segm/AP -Intact-belly_up" ]
metric_ar_pile_core_top_segm = ["crab_pile_val_core_top_instance/segm/AR1", "crab_pile_val_core_top_instance/segm/AR10", "crab_pile_val_core_top_instance/segm/AR100"]

# KeyPoints Metrics
metric_ap_easy_keypoints = ["crab_easy_val/keypoints/AP", "crab_easy_val/keypoints/AP50", "crab_easy_val/keypoints/AP75"]
metric_ap_orientation_easy_keypoints = ["crab_easy_val/keypoints/AP -Intact-belly_down", "crab_easy_val/keypoints/AP -Intact-belly_up"]
metric_ar_easy_keypoints = ["crab_easy_val/keypoints/AR", "crab_easy_val/keypoints/AR50", "crab_easy_val/keypoints/AR75"]

metric_ap_medium_keypoints = ["crab_medium_val/keypoints/AP", "crab_medium_val/keypoints/AP50", "crab_medium_val/keypoints/AP75"]
metric_ap_orientation_medium_keypoints = ["crab_medium_val/keypoints/AP -Intact-belly_down", "crab_medium_val/keypoints/AP -Intact-belly_up"]
metric_ar_medium_keypoints = ["crab_medium_val/keypoints/AR", "crab_medium_val/keypoints/AR50", "crab_medium_val/keypoints/AR75"]

metric_ap_pile_keypoints = ["crab_pile_val/keypoints/AP", "crab_pile_val/keypoints/AP50", "crab_pile_val/keypoints/AP75"]
metric_ap_orientation_pile_keypoints = ["crab_pile_val/keypoints/AP -Intact-belly_down", "crab_pile_val/keypoints/AP -Intact-belly_up"]
metric_ar_pile_keypoints = ["crab_pile_val/keypoints/AR", "crab_pile_val/keypoints/AR50", "crab_pile_val/keypoints/AR75"]

metric_ap_pile_top_keypoints = ["crab_pile_val_top_instance/keypoints/AP", "crab_pile_val_top_instance/keypoints/AP50", "crab_pile_val_top_instance/keypoints/AP75"]
metric_ap_orientation_pile_top_keypoints = ["crab_pile_val_top_instance/keypoints/AP -Intact-belly_down", "crab_pile_val_top_instance/keypoints/AP -Intact-belly_up"]
metric_ar_pile_top_keypoints = ["crab_pile_val_top_instance/keypoints/AR", "crab_pile_val_top_instance/keypoints/AR50", "crab_pile_val_top_instance/keypoints/AR75"]

metric_ap_pile_core_keypoints= ["crab_pile_val_core/keypoints/AP","crab_pile_val_core/keypoints/AP50", "crab_pile_val_core/keypoints/AP75" ] 
metric_ap_orientation_core_keypoints= ["crab_pile_val_core/keypoints/AP -Intact-belly_down","crab_pile_val_core/keypoints/AP -Intact-belly_up" ]
metric_ar_pile_core_keypoints = ["crab_pile_val_core/keypoints/AR", "crab_pile_val_core/keypoints/AR50", "crab_pile_val_core/keypoints/AR75"]

metric_ap_pile_core_top_keypoints= ["crab_pile_val_core_top_instance/keypoints/AP","crab_pile_val_core_top_instance/keypoints/AP50", "crab_pile_val_core_top_instance/keypoints/AP75" ] 
metric_ap_orientation_core_top_keypoints= ["crab_pile_val_core_top_instance/keypoints/AP -Intact-belly_down","crab_pile_val_core_top_instance/keypoints/AP -Intact-belly_up" ]
metric_ar_pile_core_top_keypoints = ["crab_pile_val_core_top_instance/keypoints/AR", "crab_pile_val_core_top_instance/keypoints/AR50", "crab_pile_val_core_top_instance/keypoints/AR75"]

# Function to find the mean values of the last 100 iterations for specified metrics  
def find_mean_last_n_metrics(data, metric_list, n=10):
    mean_values = []
    filtered_entries = [entry for entry in data if any(metric in entry for metric in metric_list)]
    last_n_entries = filtered_entries[-n:]

    for metric in metric_list:
        values = [entry.get(metric) for entry in last_n_entries]
        
        if values:
            try:
                mean_value = sum(values) / len(values)
            except:
                print("Metric:   ",metric, "Empty Values:   ", values)
            # import pdb; pdb.set_trace()
            mean_values.append({"metric": metric, "mean_value": mean_value})
        else:
            mean_values.append({"metric": metric, "mean_value": None})

    return mean_values


# Find mean values for all metrics
mean_values = []

mean_values.extend(find_mean_last_n_metrics(data, metric_ap_easy_segm + metric_ap_orientation_easy_segm + metric_ar_easy_segm))
mean_values.extend(find_mean_last_n_metrics(data, metric_ap_medium_segm + metric_ap_orientation_medium_segm + metric_ar_medium_segm))
mean_values.extend(find_mean_last_n_metrics(data, metric_ap_pile_segm + metric_ap_orientation_pile_segm + metric_ar_pile_segm))
mean_values.extend(find_mean_last_n_metrics(data, metric_ap_pile_top_segm + metric_ap_orientation_pile_top_segm + metric_ar_pile_top_segm))
mean_values.extend(find_mean_last_n_metrics(data, metric_ap_pile_core_segm + metric_ap__orientation_core_segm + metric_ar_pile_core_segm))
mean_values.extend(find_mean_last_n_metrics(data, metric_ap_pile_core_top_segm + metric_ap__orientation_core_top_segm + metric_ar_pile_core_top_segm))

mean_values.extend(find_mean_last_n_metrics(data, metric_ap_easy_keypoints + metric_ap_orientation_easy_keypoints + metric_ar_easy_keypoints))
mean_values.extend(find_mean_last_n_metrics(data, metric_ap_medium_keypoints + metric_ap_orientation_medium_keypoints + metric_ar_medium_keypoints))
mean_values.extend(find_mean_last_n_metrics(data, metric_ap_pile_keypoints + metric_ap_orientation_pile_keypoints + metric_ar_pile_keypoints))
mean_values.extend(find_mean_last_n_metrics(data, metric_ap_pile_top_keypoints + metric_ap_orientation_pile_top_keypoints + metric_ar_pile_top_keypoints))
mean_values.extend(find_mean_last_n_metrics(data, metric_ap_pile_core_keypoints + metric_ap_orientation_core_keypoints + metric_ar_pile_core_keypoints))
mean_values.extend(find_mean_last_n_metrics(data, metric_ap_pile_core_top_keypoints + metric_ap_orientation_core_top_keypoints + metric_ar_pile_core_top_keypoints))

# Convert to DataFrame and save to CSV
df = pd.DataFrame(mean_values)
output_file_path = "/home/bmv/crab_loading_cv/poet-detectron/Poet_R50_q30_RGBD_v2/RGBD_POET_mean_metrics.csv"
df.to_csv(output_file_path, index=False)

print(f"Mean metrics saved to {output_file_path}")
