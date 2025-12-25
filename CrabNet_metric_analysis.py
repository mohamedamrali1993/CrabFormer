import json
import numpy as np
import matplotlib.pyplot as plt

# Read the JSON file
file_path = "./RGBD_Q100_SwinL_batch1_iter100000/metrics.json"
with open(file_path, "r") as f:
    data = [json.loads(line) for line in f]

# Define lists of metrics
#Segm
metric_ap_easy_segm = ["crab_easy_val/segm/AP", "crab_easy_val/segm/AP50", "crab_easy_val/segm/AP75"]
metric_ap_orientation_easy_segm = ["crab_easy_val/segm/AP -Intact-belly_down", "crab_easy_val/segm/AP -Intact-belly_up"]
metric_ar_easy_segm = ["crab_easy_val/segm/AR1", "crab_easy_val/segm/AR10", "crab_easy_val/segm/AR100"]

metric_ap_medium_segm = ["crab_medium_val/segm/AP", "crab_medium_val/segm/AP50", "crab_medium_val/segm/AP75"]
metric_ap_orientation_medium_segm = ["crab_medium_val/segm/AP -Intact-belly_down", "crab_medium_val/segm/AP -Intact-belly_up"]
metric_ar_medium_segm = ["crab_medium_val/segm/AR1", "crab_medium_val/segm/AR10", "crab_medium_val/segm/AR100"]

metric_ap_pile_segm = ["crab_pile_val/segm/AP", "crab_pile_val/segm/AP50", "crab_pile_val/segm/AP75"]
metric_ap_orientation_pile_segm = ["crab_pile_val/segm/AP -Intact-belly_down", "crab_pile_val/segm/AP -Intact-belly_up"]
metric_ar_pile_segm = ["crab_pile_val/segm/AR1", "crab_pile_val/segm/AR10", "crab_pile_val/segm/AR100"]

metric_ap_pile_top_segm= ["crab_pile_val_top_instance/segm/AP","crab_pile_val_top_instance/segm/AP", "crab_pile_val_top_instance/segm/AP" ] 
metric_ap__orientation_pile_top_segm= ["crab_pile_val_top_instance/segm/AP -Intact-belly_down","crab_pile_val_top_instance/segm/AP -Intact-belly_up" ]
metric_ar_pile_top_segm = ["crab_pile_val_top_instance/segm/AR1", "crab_pile_val_top_instance/segm/AR10", "crab_pile_val_top_instance/segm/AR100"]

metric_ap_pile_core_segm= ["crab_pile_val_core/segm/AP","crab_pile_val_core/segm/AP", "crab_pile_val_core/segm/AP" ] 
metric_ap__orientation_core_segm= ["crab_pile_val_core/segm/AP -Intact-belly_down","crab_pile_val_core/segm/AP -Intact-belly_up" ]
metric_ar_pile_core_segm = ["crab_pile_val_core/segm/AR1", "crab_pile_val_core/segm/AR10", "crab_pile_val_core/segm/AR100"]

metric_ap_pile_core_top_segm= ["crab_pile_val_core_top_instance/segm/AP","crab_pile_val_core_top_instance/segm/AP", "crab_pile_val_core_top_instance/segm/AP" ] 
metric_ap__orientation_core_top_segm= ["crab_pile_val_core_top_instance/segm/AP -Intact-belly_down","crab_pile_val_core_top_instance/segm/AP -Intact-belly_up" ]
metric_ar_pile_core_top_segm = ["crab_pile_val_core_top_instance/segm/AR1", "crab_pile_val_core_top_instance/segm/AR10", "crab_pile_val_core_top_instance/segm/AR100"]



#KeyPoints
metric_ap_easy_keypoints = ["crab_easy_val/keypoints/AP", "crab_easy_val/keypoints/AP50", "crab_easy_val/keypoints/AP75"]
metric_ap_orientation_easy_keypoints = ["crab_easy_val/keypoints/AP -Intact-belly_down", "crab_easy_val/keypoints/AP -Intact-belly_up"]
metric_ar_easy_keypoints = ["crab_easy_val/keypoints/AR", "crab_easy_val/keypoints/AR50", "crab_easy_val/keypoints/AR75"]

metric_ap_medium_keypoints = ["crab_medium_val/keypoints/AP", "crab_medium_val/keypoints/AP50", "crab_medium_val/keypoints/AP75"]
metric_ap_orientation_medium_keypoints = ["crab_medium_val/keypoints/AP -Intact-belly_down", "crab_medium_val/keypoints/AP -Intact-belly_up"]
metric_ar_medium_keypoints = ["crab_medium_val/keypoints/AR", "crab_medium_val/keypoints/AR50", "crab_medium_val/keypoints/AR75"]

metric_ap_pile_keypoints = ["crab_pile_val/keypoints/AP", "crab_pile_val/keypoints/AP50", "crab_pile_val/keypoints/AP75"]
metric_ap_orientation_pile_keypoints = ["crab_pile_val/keypoints/AP -Intact-belly_down", "crab_pile_val/keypoints/AP -Intact-belly_up"]
metric_ar_pile_keypoints = ["crab_pile_val/keypoints/AR", "crab_pile_val/keypoints/AR50", "crab_pile_val/keypoints/AR75"]

metric_ap_pile_top_keypoints= ["crab_pile_val_top_instance/keypoints/AP","crab_pile_val_top_instance/keypoints/AP", "crab_pile_val_top_instance/keypoints/AP" ] 
metric_ap__orientation_pile_top_keypoints= ["crab_pile_val_top_instance/keypoints/AP -Intact-belly_down","crab_pile_val_top_instance/keypoints/AP -Intact-belly_up" ]
metric_ar_pile_top_keypoints = ["crab_pile_val_top_instance/keypoints/AR", "crab_pile_val_top_instance/keypoints/AR50", "crab_pile_val_top_instance/keypoints/AR75"]

metric_ap_pile_core_keypoints= ["crab_pile_val_core/keypoints/AP","crab_pile_val_core/keypoints/AP", "crab_pile_val_core/keypoints/AP" ] 
metric_ap__orientation_core_keypoints= ["crab_pile_val_core/keypoints/AP -Intact-belly_down","crab_pile_val_core/keypoints/AP -Intact-belly_up" ]
metric_ar_pile_core_keypoints = ["crab_pile_val_core/keypoints/AR", "crab_pile_val_core/keypoints/AR50", "crab_pile_val_core/keypoints/AR75"]

metric_ap_pile_core_top_keypoints= ["crab_pile_val_core_top_instance/keypoints/AP","crab_pile_val_core_top_instance/keypoints/AP", "crab_pile_val_core_top_instance/keypoints/AP" ] 
metric_ap__orientation_core_top_keypoints= ["crab_pile_val_core_top_instance/keypoints/AP -Intact-belly_down","crab_pile_val_core_top_instance/keypoints/AP -Intact-belly_up" ]
metric_ar_pile_core_top_keypoints = ["crab_pile_val_core_top_instance/keypoints/AR", "crab_pile_val_core_top_instance/keypoints/AR50", "crab_pile_val_core_top_instance/keypoints/AR75"]


# metric_loss= ['loss_ce', 'loss_dice' ,'loss_mask','loss_ctrs', 'loss_deltas', 'loss_kpts', 'loss_kpts_class']
# metric_total_loss = ['total_loss']
# #Class
# metric_loss_ce= ['loss_ce_0', 'loss_ce_1', 'loss_ce_2', 'loss_ce_3', 'loss_ce_4', 'loss_ce_5', 'loss_ce_6', 'loss_ce_7', 'loss_ce_8']
# #Mask
# metric_loss_dice = ['loss_dice_0', 'loss_dice_1', 'loss_dice_2', 'loss_dice_3', 'loss_dice_4', 'loss_dice_5', 'loss_dice_6', 'loss_dice_7', 'loss_dice_8']
# metric_loss_mask= ['loss_mask_0', 'loss_mask_1', 'loss_mask_2', 'loss_mask_3', 'loss_mask_4', 'loss_mask_5', 'loss_mask_6', 'loss_mask_7', 'loss_mask_8']
# #Keypoints
# metric_loss_ctrs = ['loss_ctrs_0', 'loss_ctrs_1', 'loss_ctrs_2', 'loss_ctrs_3', 'loss_ctrs_4', 'loss_ctrs_5', 'loss_ctrs_6', 'loss_ctrs_7', 'loss_ctrs_8']
# metric_loss_deltas = ['loss_deltas_0', 'loss_deltas_1', 'loss_deltas_2', 'loss_deltas_3', 'loss_deltas_4', 'loss_deltas_5', 'loss_deltas_6', 'loss_deltas_7', 'loss_deltas_8'] 
# metric_loss_kpts = ['loss_kpts_0', 'loss_kpts_1', 'loss_kpts_2', 'loss_kpts_3', 'loss_kpts_4', 'loss_kpts_5', 'loss_kpts_6', 'loss_kpts_7', 'loss_kpts_8']
# metric_loss_kpts_class = ['loss_kpts_class_0', 'loss_kpts_class_1', 'loss_kpts_class_2', 'loss_kpts_class_3', 'loss_kpts_class_4', 'loss_kpts_class_5', 'loss_kpts_class_6', 'loss_kpts_class_7', 'loss_kpts_class_8']

# validation_loss = ['validation_loss']



# Function to extract metric values
def extract_metric_values(data, metric_list):
    values = []
    for metric in metric_list:
        values.append([entry[metric] for entry in data if metric in entry])
    return values
# Function to plot metrics in subplots
def plot_multiple_metrics_subplots(metric_values_list, metric_names_list, subplot_titles):
    fig, axs = plt.subplots(1, len(metric_values_list), figsize=(18, 6), sharex=True, sharey=True)
    for i, (metric_values, metric_names, title) in enumerate(zip(metric_values_list, metric_names_list, subplot_titles)):
        for metric_value, metric_name in zip(metric_values, metric_names):
            axs[i].plot(metric_value, marker='o', linestyle='-', label=metric_name)
        axs[i].set_title(title)
        axs[i].set_xlabel('Iteration')
        axs[i].set_ylabel('Metric Value (%)')
        axs[i].legend()
        axs[i].grid(True)
        axs[i].set_yticks(np.arange(0, 101, 10))  # Set y-axis ticks from 0 to 100 in steps of 10
    plt.tight_layout()
    plt.show()

def plot_metrics(metric_values_list, metric_names_list, plot_title):
    plt.figure(figsize=(12, 6))
    for metric_values, metric_names in zip(metric_values_list, metric_names_list):
        for metric_value, metric_name in zip(metric_values, metric_names):
            plt.plot(metric_value, marker='o', linestyle='-', label=metric_name)
    plt.title(plot_title)
    plt.xlabel('Iteration')
    plt.ylabel('Metric Value (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Extract metric values
#Segm
metric_ap_values_easy_segm = extract_metric_values(data, metric_ap_easy_segm)
metric_ap_orientation_values_easy_segm = extract_metric_values(data, metric_ap_orientation_easy_segm)
metric_ar_values_easy_segm = extract_metric_values(data, metric_ar_easy_segm)

metric_ap_values_medium_segm = extract_metric_values(data, metric_ap_medium_segm)
metric_ap_orientation_values_medium_segm = extract_metric_values(data, metric_ap_orientation_medium_segm)
metric_ar_values_medium_segm = extract_metric_values(data, metric_ar_medium_segm)

metric_ap_values_pile_segm = extract_metric_values(data, metric_ap_pile_segm)
metric_ap_orientation_values_pile_segm = extract_metric_values(data, metric_ap_orientation_pile_segm)
metric_ar_values_pile_segm = extract_metric_values(data, metric_ar_pile_segm)

metric_ap_values_pile_top_segm = extract_metric_values(data, metric_ap_pile_top_segm)
metric_ap_orientation_values_pile_top_segm = extract_metric_values(data, metric_ap__orientation_pile_top_segm)
metric_ar_values_pile_top_segm = extract_metric_values(data, metric_ar_pile_top_segm)

metric_ap_values_pile_core_segm = extract_metric_values(data, metric_ap_pile_core_segm)
metric_ap_orientation_values_pile_core_segm = extract_metric_values(data, metric_ap__orientation_core_segm)
metric_ar_values_pile_core_segm = extract_metric_values(data, metric_ar_pile_core_segm)

metric_ap_values_pile_core_top_segm = extract_metric_values(data, metric_ap_pile_core_top_segm)
metric_ap_orientation_values_pile_core_top_segm = extract_metric_values(data, metric_ap__orientation_core_segm)
metric_ar_values_pile_core_top_segm = extract_metric_values(data, metric_ar_pile_core_segm)

#KeyPoints
metric_ap_values_easy_keypoints = extract_metric_values(data, metric_ap_easy_keypoints)
metric_ap_orientation_values_easy_keypoints = extract_metric_values(data, metric_ap_orientation_easy_keypoints)
metric_ar_values_easy_keypoints = extract_metric_values(data, metric_ar_easy_keypoints)

metric_ap_values_medium_keypoints = extract_metric_values(data, metric_ap_medium_keypoints)
metric_ap_orientation_values_medium_keypoints = extract_metric_values(data, metric_ap_orientation_medium_keypoints)
metric_ar_values_medium_keypoints = extract_metric_values(data, metric_ar_medium_keypoints)

metric_ap_values_pile_keypoints = extract_metric_values(data, metric_ap_pile_keypoints)
metric_ap_orientation_values_pile_keypoints = extract_metric_values(data, metric_ap_orientation_pile_keypoints)
metric_ar_values_pile_keypoints = extract_metric_values(data, metric_ar_pile_keypoints)

metric_ap_values_pile_top_keypoints= extract_metric_values(data, metric_ap_pile_top_keypoints)
metric_ap_orientation_values_pile_top_keypoints = extract_metric_values(data, metric_ap__orientation_pile_top_keypoints)
metric_ar_values_pile_top_keypoints = extract_metric_values(data, metric_ar_pile_top_keypoints)

metric_ap_values_pile_core_keypoints= extract_metric_values(data, metric_ap_pile_core_keypoints)
metric_ap_orientation_values_pile_core_keypoints = extract_metric_values(data, metric_ap__orientation_core_keypoints)
metric_ar_values_pile_core_keypoints = extract_metric_values(data, metric_ar_pile_core_keypoints)

metric_ap_values_pile_core_top_keypoints= extract_metric_values(data, metric_ap_pile_core_top_keypoints)
metric_ap_orientation_values_pile_core_top_keypoints = extract_metric_values(data, metric_ap__orientation_core_top_keypoints)
metric_ar_values_pile_core_top_keypoints = extract_metric_values(data, metric_ar_pile_core_top_keypoints)

#Losses
# metric_loss_values = extract_metric_values(data, metric_loss)
# metric_total_loss_values = extract_metric_values(data, metric_total_loss)
# metric_loss_ce_values = extract_metric_values(data, metric_loss_ce)
# metric_loss_dice_values = extract_metric_values(data, metric_loss_dice)
# metric_loss_mask_values = extract_metric_values(data, metric_loss_mask)
# metric_loss_ctrs_values = extract_metric_values(data, metric_loss_ctrs)
# metric_loss_deltas_values = extract_metric_values(data, metric_loss_deltas)
# metric_loss_kpts_values = extract_metric_values(data, metric_loss_kpts)
# metric_loss_kpts_class_values = extract_metric_values(data, metric_loss_kpts_class)
# validation_loss_values = extract_metric_values(data, validation_loss)


# Plot metrics in subplots
#Segm
plot_multiple_metrics_subplots([metric_ap_values_easy_segm, metric_ap_orientation_values_easy_segm, metric_ar_values_easy_segm],
                        [metric_ap_easy_segm, metric_ap_orientation_easy_segm, metric_ar_easy_segm],
                        ['AP (Easy Case Segmentation)', 'AP Orientation (Easy Case Segmentation)', 'AR (Easy Case Segmentation)'])

plot_multiple_metrics_subplots([metric_ap_values_medium_segm, metric_ap_orientation_values_medium_segm, metric_ar_values_medium_segm],
                        [metric_ap_medium_segm, metric_ap_orientation_medium_segm, metric_ar_medium_segm],
                        ['AP (Medium Case Segmentation)', 'AP Orientation (Medium Case Segmentation)', 'AR (Medium Case Segmentation)'])

plot_multiple_metrics_subplots([metric_ap_values_pile_segm, metric_ap_orientation_values_pile_segm, metric_ar_values_pile_segm],
                        [metric_ap_pile_segm, metric_ap_orientation_pile_segm, metric_ar_pile_segm],
                        ['AP (Pile Case Segmentation)', 'AP Orientation (Pile Case Segmentation)', 'AR (Pile Case Segmentation)'])

plot_multiple_metrics_subplots([metric_ap_values_pile_top_segm, metric_ap_orientation_values_pile_top_segm, metric_ar_values_pile_top_segm],
                        [metric_ap_pile_top_segm, metric_ap__orientation_pile_top_segm, metric_ar_pile_top_segm],
                        ['AP (Pile Top Case Segmentation)', 'AP Orientation (Pile Top Case Segmentation)', 'AR (Pile Top Case Segmentation)'])

plot_multiple_metrics_subplots([metric_ap_values_pile_core_segm, metric_ap_orientation_values_pile_core_segm, metric_ar_values_pile_core_segm],
                        [metric_ap_pile_core_segm, metric_ap__orientation_core_segm, metric_ar_pile_core_segm],
                        ['AP (Pile Core Case Segmentation)', 'AP Orientation (Pile Core Case Segmentation)', 'AR (Pile Core Case Segmentation)'])

plot_multiple_metrics_subplots([metric_ap_values_pile_core_top_segm, metric_ap_orientation_values_pile_core_top_segm, metric_ar_values_pile_core_top_segm],
                        [metric_ap_pile_core_top_segm, metric_ap__orientation_core_segm, metric_ar_pile_core_segm],
                        ['AP (Pile Core Top Case Segmentation)', 'AP Orientation (Pile Core Top Case Segmentation)', 'AR (Pile Core Top Case Segmentation)'])

#KeyPoints
plot_multiple_metrics_subplots([metric_ap_values_easy_keypoints, metric_ap_orientation_values_easy_keypoints, metric_ar_values_easy_keypoints],
                        [metric_ap_easy_keypoints, metric_ap_orientation_easy_keypoints, metric_ar_easy_keypoints],
                        ['AP (Easy Case Keypoints)', 'AP Orientation (Easy Case Keypoints)', 'AR (Easy Case Keypoints)'])

plot_multiple_metrics_subplots([metric_ap_values_medium_keypoints, metric_ap_orientation_values_medium_keypoints, metric_ar_values_medium_keypoints],
                        [metric_ap_medium_keypoints, metric_ap_orientation_medium_keypoints, metric_ar_medium_keypoints],
                        ['AP (Medium Case Keypoints)', 'AP Orientation (Medium Case Keypoints)', 'AR (Medium Case Keypoints)'])

plot_multiple_metrics_subplots([metric_ap_values_pile_keypoints, metric_ap_orientation_values_pile_keypoints, metric_ar_values_pile_keypoints],
                        [metric_ap_pile_keypoints, metric_ap_orientation_pile_keypoints, metric_ar_pile_keypoints],
                        ['AP (Pile Case Keypoints)', 'AP Orientation (Pile Case Keypoints)', 'AR (Pile Case Keypoints)'])

plot_multiple_metrics_subplots([metric_ap_values_pile_top_keypoints, metric_ap_orientation_values_pile_top_keypoints, metric_ar_values_pile_top_keypoints],
                        [metric_ap_pile_top_keypoints, metric_ap__orientation_pile_top_keypoints, metric_ar_pile_top_keypoints],
                        ['AP (Pile Top Case Keypoints)', 'AP Orientation (Pile Top Case Keypoints)', 'AR (Pile Top Case Keypoints)'])

plot_multiple_metrics_subplots([metric_ap_values_pile_core_keypoints, metric_ap_orientation_values_pile_core_keypoints, metric_ar_values_pile_core_keypoints],
                        [metric_ap_pile_core_keypoints, metric_ap__orientation_core_keypoints, metric_ar_pile_core_keypoints],
                        ['AP (Pile Core Case Keypoints)', 'AP Orientation (Pile Core Case Keypoints)', 'AR (Pile Core Case Keypoints)'])

plot_multiple_metrics_subplots([metric_ap_values_pile_core_top_keypoints, metric_ap_orientation_values_pile_core_top_keypoints, metric_ar_values_pile_core_top_keypoints],
                        [metric_ap_pile_core_top_keypoints, metric_ap__orientation_core_keypoints, metric_ar_pile_core_keypoints],
                        ['AP (Pile Core Top Case Keypoints)', 'AP Orientation (Pile Core Top Case Keypoints)', 'AR (Pile Core Top Case Keypoints)'])


#Losses
# plot_metrics([metric_total_loss_values], [metric_total_loss] , ['Total Loss'])
# plot_metrics([metric_loss_values], [metric_loss] , ['Metric Loss'])
# plot_metrics([metric_loss_ce_values], [metric_loss_ce] , ['Loss CE'])
# plot_metrics([metric_loss_dice_values], [metric_loss_dice] , ['Loss Dice'])
# plot_metrics([metric_loss_mask_values], [metric_loss_mask] , ['Loss Mask'])
# plot_metrics([metric_loss_ctrs_values], [metric_loss_ctrs] , ['Loss CTRS'])
# plot_metrics([metric_loss_deltas_values], [metric_loss_deltas] , ['Loss Deltas'])
# plot_metrics([metric_loss_kpts_values], [metric_loss_kpts] , ['Loss KPTS'])
# plot_metrics([metric_loss_kpts_class_values], [metric_loss_kpts_class] , ['Loss KPTS Class'])
# plot_metrics([validation_loss_values], [validation_loss] , ['Validation Loss'])



