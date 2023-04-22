import json
import numpy as np
import os
import pandas as pd
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_dilation
import argparse
import re
import csv
import random
from collections import defaultdict
from pprint import pprint

def shift_points(points):
    # Shift points according to the padding we apply to the masks in backproject_masks_new.py
    shifted_points = []
    for y, x in points:
        if x >= 1750:
            x -= 1750
        else:
            x += 250
        shifted_points.append((y, x))
    return shifted_points

def get_ps_labels(args):

    base_url = "https://sidewalk-amsterdam.cs.washington.edu/v2/access/attributesWithLabels?lat1={}&lng1={}&lat2={}&lng2={}" 
    
    whole = (52.303, 4.8, 52.425, 5.05)
    centrum_west = (52.364925, 4.87444, 52.388692, 4.90641)
    test = (52.0, 4.0, 53.0, 5.0)

    coords = test

    url = base_url.format(*coords)

    label_dump = os.path.join(args.label_dump, 'attributesWithLabels')

    try:
        project_sidewalk_labels = json.load(open(label_dump, 'r'))
    except Exception as e:
        print("Couldn't load local dump")
        project_sidewalk_labels = requests.get(url.format(*coords)).json()
        json.dump(project_sidewalk_labels, open(label_dump, 'w'))

    ps_labels_df = gpd.GeoDataFrame.from_features(project_sidewalk_labels['features'])
    # Print length before filtering
    print('Length of labels from attributesWithLabels API: ', len(ps_labels_df))

    return ps_labels_df

def get_xy_coords_ps_labels():
    # Send a get call to this API: https://sidewalk-amsterdam-test.cs.washington.edu/adminapi/labels/cvMetadata
    other_labels = requests.get('https://sidewalk-amsterdam.cs.washington.edu/adminapi/labels/cvMetadata').json()
    other_labels_df = pd.DataFrame(other_labels)
    print('Length raw labels from cvMetadata API (low quality) : ', len(other_labels_df))

    return other_labels_df

def visualize_labels(args, gt_points, pano_id, path):
    # Load the image
    image_path = os.path.join(args.input_dir, pano_id)
    backprojected_path = os.path.join(os.path.dirname(args.input_dir), args.backprojected_dir)
    mask_path = os.path.join(backprojected_path, f'{pano_id}/{pano_id}.png')
    # Depending on the machine, the path might be different
    # Try to load the image with and without the .jpg extension
    try:
        image = plt.imread(image_path + '.jpg')
    except:
        image = plt.imread(image_path)
    mask = plt.imread(mask_path)

    # Pad the image to correctly visualize the mask
    image = np.concatenate((image[:, 1750:], image[:, :1750]), axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    ax.imshow(mask, cmap='jet', alpha=0.4)

    for gt_point in gt_points:
        ax.scatter(gt_point[1], gt_point[0], color='red', s=7)

    # Set small tick labels
    ax.tick_params(axis='both', which='both', labelsize=6)
    # No ticks
    ax.set_xticks([])

    # Add legend for red points
    ax.scatter([], [], color='red', s=8, label='Ground Truth')
    ax.legend(prop={'size': 5})

    # Save the image
    fig.savefig(os.path.join(path, f'{pano_id}.png'), dpi=300, bbox_inches='tight')
    # Close the figure
    plt.close(fig)


def visualize_debug_mask(gt_points, pred_masks, closest_points, gt_indices, pano_id, path, cp=True):
    num_masks = len(pred_masks)
    num_rows = int(np.ceil(num_masks / 2))
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 3 * num_rows))

    assert len(pred_masks) == len(closest_points) == len(gt_indices)

    # Remove space between rows and adjust the space between title and first row
    fig.subplots_adjust(hspace=0, top=0.9)

    for idx, (pred_mask, (row_idx, col_idx)) in enumerate(zip(pred_masks, np.ndindex((num_rows, 2)))):
        if num_rows > 1:
            ax = axes[row_idx, col_idx]
        else:
            ax = axes[col_idx]
        ax.imshow(pred_mask)  # Display the predicted mask

        # Retrieve the correct ground truth point, distance, and closest point
        gt_point = gt_points[gt_indices[idx]]
        closest_point = closest_points[idx]

        y, x = gt_point
        closest_y, closest_x = closest_point

        ax.scatter(x, y, c='red', marker='x', s=20, label='Ground Truth')  # Mark the ground truth point
        ax.plot([x, closest_x], [y, closest_y], 'r--', label='Distance')  # Draw the distance line
        ax.scatter(closest_x, closest_y, c='blue', marker='o', s=20, label='Closest Point')  # Mark the closest point

        # Put legend size to small
        ax.legend(loc='upper right', prop={'size': 7})
        ax.set_title(f"Closest mask-to-point distance for mask {idx + 1}", fontsize=7)

        # Display y-axis only on the left-most plot for each row
        if col_idx != 0:
            ax.set_yticklabels([])
            ax.set_yticks([])

    # Make ticks smaller
    for ax in fig.get_axes():
        ax.tick_params(axis='both', which='both', labelsize=7)

    # Remove any unused plot axes
    for i in range(num_rows * 2 - num_masks):
        if num_rows > 1:
            fig.delaxes(axes[num_rows - 1, 1 - i])
        else:
            fig.delaxes(axes[1 - i])

    plt.tight_layout()
    #plt.show()

    if cp:
        filename = f'{pano_id}_mask_cp.png'
    else:
        filename = f'{pano_id}_mask_ap.png'

    fig.savefig(os.path.join(path, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_best_dilated(args, gt_points, pred_masks, best_gt_point_indices, pano_id, path):
    num_masks = len(pred_masks)
    num_rows = int(np.ceil(num_masks / 2))
    fig, axes = plt.subplots(num_rows, 2, figsize=(13, 4 * num_rows))

    assert len(pred_masks) == len(best_gt_point_indices)

    # Remove space between rows and adjust the space between title and first row
    fig.subplots_adjust(hspace=0, top=0.9)

    # Generate a grid of coordinates for the entire image
    y_coords, x_coords = np.indices(pred_masks[0].shape)
    coords = np.column_stack((y_coords.ravel(), x_coords.ravel()))

    for idx, (pred_mask, best_gt_point_idx) in enumerate(zip(pred_masks, best_gt_point_indices)):
        best_gt_point = gt_points[best_gt_point_idx]

        # Compute the distance from the best ground truth point to all other points
        distances = cdist([best_gt_point], coords).reshape(pred_mask.shape)

        # Create a dilated mask by thresholding the distance matrix at the specified radius
        gt_mask_dilated = (distances <= args.radius).astype(int)

        # Determine the position of the current mask in the plot grid
        row_idx = idx // 2
        col_idx = idx % 2

        # Access the axes based on its dimensions
        if num_rows > 1:
            ax = axes[row_idx, col_idx]
        else:
            ax = axes[col_idx]

        # Display the dilated ground truth mask
        ax.imshow(gt_mask_dilated, alpha=0.5, cmap='gray', label='Dilated Ground Truth')

        # Display the predicted mask
        ax.imshow(pred_mask, alpha=0.5, cmap='jet', label='Predicted Mask')

        # Mark the best ground truth point
        ax.scatter(best_gt_point[1], best_gt_point[0], c='red', marker='x', s=50, label='Ground Truth')

        # Set the plot title
        ax.set_title(f"Mask {idx + 1}", fontsize='medium')
        ax.legend()

        # Display y-axis only on the left-most plot for each row
        if col_idx != 0:
            ax.set_yticklabels([])
            ax.set_yticks([])

    # Remove any unused plot axes
    for i in range(num_rows * 2 - num_masks):
        if num_rows > 1:
            fig.delaxes(axes[num_rows - 1, 1 - i])
        else:
            fig.delaxes(axes[1 - i])

    # Set spacing between rows
    fig.subplots_adjust(hspace=0.1)

    # Use tight_layout with rect parameter to align the title properly
    plt.tight_layout()

    # Save the image
    fig.savefig(os.path.join(path, f'{pano_id}_mask_iou.png'), dpi=300, bbox_inches='tight')
    # Close the figure
    plt.close(fig)


def compute_label_coordinates(args, dataframe, pano_id):            

    # Load pano
    image_path = os.path.join(args.input_dir,pano_id)
    # Depending on the machine, the path might be different
    # Try to load the image with and without the .jpg extension
    try:
        image = plt.imread(image_path + '.jpg')
    except:
        image = plt.imread(image_path)

    # Find all the labels in the panorama
    labels = dataframe[dataframe['gsv_panorama_id'] == pano_id]

    labels_coords = []

    # For each label, find its coordinates in the panorama
    for index, row in labels.iterrows():
        #print('Labels coming from panorama:', row['gsv_panorama_id'])
        label_name = row['label_id']
        image_width = row['pano_width']
        image_height = row['pano_height']
        sv_image_x = row['pano_x']
        # Momentarily change the definition of sv_image_y. For now,
        # sv_image_y is the offset w.r.t. the middle of the image
        # which means that the real y coordinate is y = (image_width / 2) - sv_image_y
        sv_image_y = row['pano_y']
        #real_y = (image_height / 2) - sv_image_y

        #print(f'Original image width: {image_width}, height: {image_height}')
        #print(f'Original label {label_name} coordinates: {sv_image_x}, {sv_image_y}')
        #print(f'Original label {label_name} with real y-coordinates' \
        #    f'({image_height}/2){sv_image_y}: {sv_image_x}, {real_y}')

        # image_width : sv_image_x = my_pano_width : sv_pano_x
        # sv_pano_x = (sv_image_x * my_pano_width) / image_width
        sv_pano_x = int(sv_image_x*image.shape[1]/image_width)
        sv_pano_y = int(sv_image_y*image.shape[0]/image_height)

        #print(f'Scaled label {label_name} coordinates: {sv_pano_x}, {sv_pano_y} \n')

        # Visualize the point (sv_pano_x, sv_pano_y) on the image
        #print(f'Label {label_name} coordinates: {sv_pano_x}, {sv_pano_y} \n')
        #plt.scatter(sv_pano_x, sv_pano_y, color='red', s=10)

        labels_coords.append((sv_pano_y, sv_pano_x))

    return labels_coords

import numpy as np
from scipy.spatial.distance import cdist

def mask_to_point_distance(gt_points, pred_masks, closest_point=True):
    distances = []
    points = []
    gt_indices = []
    is_empty = False

    # Function to determine the horizontal part of the image
    def get_image_part(coord):
        if 0 <= coord < 500:
            return 0
        elif 500 <= coord < 1000:
            return 1
        elif 1000 <= coord < 1500:
            return 2
        elif 1500 <= coord < 2000:
            return 3
        else:
            raise ValueError("Invalid coordinate")

    for idx, pred_mask in enumerate(pred_masks):
        if pred_mask.ndim == 2:  # Check if the mask is 2D
            mask_channel = pred_mask
        elif pred_mask.ndim == 3:  # Check if the mask is 3D
            mask_channel = pred_mask[:, :, 0]  # Assuming the first channel contains the binary mask
        else:
            raise ValueError("Invalid mask dimensions")

        mask_coords = np.where(mask_channel > 0)  # Get the indices of non-zero elements (i.e., the mask)
        mask_coords = np.vstack(mask_coords).T  # Stack the mask indices into a 2D array

        if closest_point:
            min_distance = float('inf')
            closest_y, closest_x = -1, -1
            gt_point_index = -1

            for gt_idx, gt_point in enumerate(gt_points):
                point_coords = np.array(gt_point).reshape(1, -1)
                dist = cdist(point_coords, mask_coords)

                try:
                    local_min_distance_idx = np.argmin(dist)
                except:
                    print('Mask is empty.')
                    is_empty = True
                    break
                local_min_distance = dist[0, local_min_distance_idx]
                local_closest_y, local_closest_x = mask_coords[local_min_distance_idx]

                gt_part = get_image_part(gt_point[1])
                local_closest_x_part = get_image_part(local_closest_x)

                if local_min_distance < min_distance and gt_part == local_closest_x_part:
                    min_distance = local_min_distance
                    closest_y, closest_x = local_closest_y, local_closest_x
                    gt_point_index = gt_idx

            distances.append(min_distance)
            points.append((closest_y, closest_x))
            gt_indices.append(gt_point_index)

        else:
            # Compute the anchor point as the centroid of the mask coordinates
            anchor_point = np.mean(mask_coords, axis=0)

            min_distance = float('inf')
            gt_point_index = -1

            for gt_idx, gt_point in enumerate(gt_points):
                point_coords = np.array(gt_point).reshape(1, -1)
                dist = cdist(point_coords, anchor_point.reshape(1, -1))  # Calculate distance to anchor_point

                local_min_distance = dist[0, 0]

                gt_part = get_image_part(gt_point[1])
                anchor_point_part = get_image_part(anchor_point[1])

                if local_min_distance < min_distance and gt_part == anchor_point_part:
                    min_distance = local_min_distance
                    gt_point_index = gt_idx

            distances.append(min_distance)
            points.append(anchor_point)
            gt_indices.append(gt_point_index)

    return distances, points, gt_indices, is_empty

def mask_to_point_best_iou(gt_points, pred_masks, radius=5):
    best_ious = []
    best_gt_point_indices = []

    # Generate a grid of coordinates for the entire image
    y_coords, x_coords = np.indices(pred_masks[0].shape)
    coords = np.column_stack((y_coords.ravel(), x_coords.ravel()))

    # Function to determine the horizontal part of the image
    def get_image_part(coord):
        if 0 <= coord < 500:
            return 0
        elif 500 <= coord < 1000:
            return 1
        elif 1000 <= coord < 1500:
            return 2
        elif 1500 <= coord < 2000:
            return 3
        else:
            raise ValueError("Invalid coordinate")

    for idx, pred_mask in enumerate(pred_masks):
        max_iou = -1  # Initialize with -1 to distinguish cases when no IoU is calculated
        best_gt_point_index = -1

        # Get a random point in the mask and find its image part
        mask_points = np.where(pred_mask)
        random_index = random.randint(0, len(mask_points[0]) - 1)
        random_mask_point = (mask_points[0][random_index], mask_points[1][random_index])
        mask_part = get_image_part(random_mask_point[1])

        for point_idx, gt_point in enumerate(gt_points):
            gt_part = get_image_part(gt_point[1])

            # Check if the mask and the label are both in the same part
            if mask_part == gt_part:
                print(f'Mask and Ground Truth label are in the same part: {gt_part} == {mask_part}')

                # Compute the distance from the ground truth point to all other points
                distances = cdist([gt_point], coords).reshape(pred_mask.shape)

                # Create a dilated mask by thresholding the distance matrix at the specified radius
                gt_mask_dilated = (distances <= radius).astype(int)

                # Calculate IoU
                intersection = np.sum(np.logical_and(gt_mask_dilated, pred_mask))
                union = np.sum(np.logical_or(gt_mask_dilated, pred_mask))
                iou = intersection / union

                # Update the best IoU and ground truth point index
                if iou > max_iou:
                    max_iou = iou
                    best_gt_point_index = point_idx

        best_ious.append(max_iou)
        best_gt_point_indices.append(best_gt_point_index)

    return best_ious, best_gt_point_indices

def precision_recall_f1(distances, gt_indices, threshold):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    assert len(distances) == len(gt_indices)

    matched_gt_indices = set()

    # Count true positives and false positives
    for distance, gt_idx in zip(distances, gt_indices):
        if distance <= threshold:
            true_positives += 1
            matched_gt_indices.add(gt_idx)
        else:
            false_positives += 1

    # Count false negatives
    false_negatives = len(gt_indices) - len(matched_gt_indices)

    '''True Negative (TN) — Background region correctly not detected by the model. 
    This metric is not used in object detection because such regions are not explicitly
    annotated when preparing the annotations.'''

    # Deal with division by zero exception (true_positives + false_positives == 0 or true_positives + false_negatives == 0)
    if true_positives + false_positives == 0:
        precision = 0
        recall = 0
        f1_score = 0
    elif true_positives + false_negatives == 0:
        precision = 0
        recall = 0
        f1_score = 0
    else:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    
    #print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1_score:.3f}")
    return precision, recall, f1_score

def average_precision(distances, gt_indices, threshold):
    true_positives = 0
    false_positives = 0
    n_gt_points = len(gt_indices)

    assert len(distances) == len(gt_indices)

    matched_gt_indices = set()

    # Count true positives and false positives
    for distance, gt_idx in zip(distances, gt_indices):
        if distance <= threshold:
            true_positives += 1
            matched_gt_indices.add(gt_idx)
        else:
            false_positives += 1

    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / n_gt_points

    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    i = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])

    return ap

def average_precision_iou(best_ious, best_gt_point_indices, threshold):
    num_gt_points = len(best_gt_point_indices)
    num_pred_masks = len(best_ious)

    assert num_gt_points == num_pred_masks

    sorted_indices = sorted(range(num_pred_masks), key=lambda i: best_ious[i], reverse=True)
    sorted_best_ious = [best_ious[i] for i in sorted_indices]

    tp = [1 if iou >= threshold else 0 for iou in sorted_best_ious]
    fp = [1 - t for t in tp]

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / num_gt_points

    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    i = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])

    return ap

def evaluate_single_batch(args, batch, other_labels_df, directory):
    panos = {}
    for instance in batch:
        pano_id = instance['pano_id'].replace(".jpg", "")
        mask = mask_util.decode(instance['segmentation'])
        if pano_id not in panos:
            panos[pano_id] = []
        panos[pano_id].append(mask)

    for pano in panos:
        '''Assumptions: there is one ground truth for each mask, 
        and the order of the masks is the same as the order of the ground truth indices.'''
        if pano in other_labels_df["gsv_panorama_id"].values:
            print(f"Evaluating {pano}...")
            pred_masks = panos[pano]
            n_masks = len(panos[pano])

            # Compute label coordinates for pano
            gt_points = compute_label_coordinates(args, other_labels_df, pano)

            # Since we padded the masks, we need to apply the same padding to the ground truth points
            gt_points = shift_points(gt_points)
            

            # Compute metrics
            # Metrics 1: (average) mask-to-closest-point distance
            cp_distances, closest_points, cp_gt_indices, is_empty = mask_to_point_distance(gt_points, pred_masks, True)
            if is_empty:
                continue
            # Calculate mean distance between all masks and points, closest points only
            # Exclude float('inf') values
            cp_distances_array = np.array(cp_distances)
            non_inf_indices = np.where(np.logical_not(np.isinf(cp_distances_array)))
            cp_distances_filtered = cp_distances_array[non_inf_indices]
            cp_mean_distance = np.mean(cp_distances_filtered)
            # Metrics 1.1: (average) mask-to-closest-anchor-point distance
            ap_distances, closest_anchors, ap_gt_indices, is_empty = mask_to_point_distance(gt_points, pred_masks, False)
            ap_distances_array = np.array(ap_distances)
            non_inf_ap_indices = np.where(np.logical_not(np.isinf(ap_distances_array)))
            ap_distances_filtered = ap_distances_array[non_inf_ap_indices]
            ap_mean_distance = np.mean(ap_distances_filtered)
            # Metrics 2: point-to-mask best IoU
            # Exclude -1 values )
            best_ious, best_gt_point_indices = mask_to_point_best_iou(gt_points, pred_masks, radius=100)
            best_ious_array = np.array(best_ious)
            non_minus_one_indices = np.where(best_ious_array != -1)
            best_ious_filtered = best_ious_array[non_minus_one_indices]
            mean_ious = np.mean(best_ious_filtered)

            # Filter valid quadruples based on gt_indices
            valid_cp_quadruples = [(m, d, p, i) for m, d, p, i in zip(pred_masks, cp_distances, closest_points, cp_gt_indices) if i != -1]
            valid_ap_quadruples = [(m, d, p, i) for m, d, p, i in zip(pred_masks, ap_distances, closest_anchors, ap_gt_indices) if i != -1]
            valid_iou_triples = [(m, iou, i) for m, iou, i in zip(pred_masks, best_ious, best_gt_point_indices) if i != -1]

            # Extract valid masks, distances, points, and gt_indices
            valid_cp_masks, valid_cp_distances, valid_closest_points, valid_cp_gt_indices = zip(*valid_cp_quadruples)
            valid_ap_masks, valid_ap_distances, valid_closest_anchors, valid_ap_gt_indices = zip(*valid_ap_quadruples)
            valid_iou_masks, valid_best_ious, valid_best_gt_point_indices = zip(*valid_iou_triples)

            # Metrics 3: Precision, Recall, F1 (already uses closest_points based on point-to-mask distance)
            cp_precision, cp_recall, cp_f1 = precision_recall_f1(valid_cp_distances, valid_cp_gt_indices, args.threshold)
            # Metrics 3.1: Precision, Recall, F1 (already uses anchor_points based on point-to-mask distance)
            ap_precision, ap_recall, ap_f1 = precision_recall_f1(valid_ap_distances, valid_ap_gt_indices, args.threshold)
            # Metrics 4: Average precision based on point-to-mask distance
            cp_ap = average_precision(valid_cp_distances, valid_cp_gt_indices, args.threshold)
            # Metrics 4.1: Average precision based on point-to-mask distance
            ap_ap = average_precision(valid_ap_distances, valid_ap_gt_indices, args.threshold)
            # Metrics 5: Average precision @50 and @75 based on IoU
            ap_50 = average_precision_iou(valid_best_ious, valid_best_gt_point_indices, threshold=0.5)
            ap_75 = average_precision_iou(valid_best_ious, valid_best_gt_point_indices, threshold=0.75)
                    
            # Save metrics to .csv file, adding as first column the panorama ID. Add header
            metrics = [pano, cp_mean_distance, ap_mean_distance, mean_ious, \
                       cp_precision, ap_precision, cp_recall, ap_recall, \
                        cp_f1, ap_f1, cp_ap, ap_ap, ap_50, ap_75]
            with open(os.path.join(directory, f'metrics_{args.threshold}_{args.radius}.csv'), 'a') as f:
                writer = csv.writer(f)
                # If the first row contains the header, don't write it again
                if os.stat(os.path.join(directory, f'metrics_{args.threshold}_{args.radius}.csv')).st_size == 0:
                    writer.writerow(['pano_id', 'avg_cp_distance', 'avg_ap_distance', \
                                    'avg_iou', 'cp_precision', 'ap_precision', \
                                    'cp_recall', 'ap_recall', 'cp_f1', 'ap_f1', \
                                    'cp_ap', 'ap_ap', 'ap50', 'ap75'])
                writer.writerow(metrics)

            if args.visualize:
                # Probability of visualization: 10% of the panos
                if random.random() >= 0.1:
                    continue
                else:
                    visualization_path = os.path.join(directory, 'visualized')
                    print(f'Visualization of pano {pano} with threshold=', args.threshold, ' and radius=', args.radius, '')
                    # Make a folder 'visualized' if not present
                    if not os.path.exists(visualization_path):
                        os.makedirs(visualization_path)
                    # Make a folder for the pano
                    pano_path = os.path.join(visualization_path, pano)
                    if not os.path.exists(pano_path):
                        os.makedirs(pano_path)
                    visualize_labels(args, gt_points, pano, pano_path) # visualize labels and masks on pano
                    visualize_debug_mask(gt_points, valid_cp_masks, \
                                        valid_closest_points, valid_cp_gt_indices, pano, pano_path, True) # metrics 1
                    visualize_debug_mask(gt_points, valid_ap_masks, \
                                        valid_closest_anchors, valid_ap_gt_indices, pano, pano_path, False) # metrics 1.1
                    visualize_best_dilated(args, gt_points, valid_iou_masks, valid_best_gt_point_indices, \
                                            pano, pano_path) # metrics 2

def evaluate(args, directory):

    # Set a seed to reproduce the same randomness of panos visualization
    random.seed(42)

    # Get Project Sidewalk labels from API
    ps_labels_df = get_ps_labels(args)
    # Get the labels from another API endpoint
    other_labels_df = get_xy_coords_ps_labels()

    # Filter other_labels_df to only contain obstacles ('label_type_id' == 3)
    other_labels_df = other_labels_df[other_labels_df['label_type_id'] == 3]

    print(f'Number of obstacles labels from cvMetadata API (low quality): {len(other_labels_df)}')

    # Intersect other_labels_df with ps_labels_df
    other_labels_df = other_labels_df[other_labels_df['label_id'].isin(ps_labels_df['label_id'])]
    print(f'Number of obstacles labels in other_labels_df after filtering for high quality data: {len(other_labels_df)}')

    # Load masks from .json file
    json_path = os.path.join(os.path.dirname(args.input_dir), args.backprojected_dir)
    json_path = os.path.join(json_path, 'input_coco_format.json')
    with open(json_path) as f:
        data = json.load(f)

    # The structure of data is a list of dictionaries, with instance['pano_id'] as the panorama ID
    # and instance['segmentation'] as the corresponding masks.
    # We want to group the instances by pano_id, and then split the data into batches
    # with no more than 5 different pano_ids, so that the CPU can handle it.

    # Group instances by pano_id
    grouped_data = defaultdict(list)
    # Count instances in data
    count = 0
    for instance in data:
        count += 1
        grouped_data[instance['pano_id']].append(instance)

    print(f'Number of instances (masks): {count}')
    print(f'Number of panos: {len(grouped_data)}')

    # Split data into batches with no more than 5 different pano_ids
    batches = []
    batch = []
    pano_count = 0

    for instances in grouped_data.values():
        if pano_count < 5:
            batch.extend(instances)
            pano_count += 1
        else:
            batches.append(batch)
            batch = instances.copy()
            pano_count = 1

        if pano_count == 5:
            batches.append(batch)
            batch = []
            pano_count = 0

    if batch:
        batches.append(batch)

    print(f'Number of batches: {len(batches)}')

    if batch: batches.append(batch)

    for data_batch in tqdm(batches):
        # Check if cpu can handle the batch
        evaluate_single_batch(args, data_batch, other_labels_df, directory)
            
    # Open the .csv file, make it a dataframe, calculate the average of each column,
    # and save it to a new .csv file
    metrics_df = pd.read_csv(os.path.join(directory, f'metrics_{args.threshold}_{args.radius}.csv'))

    # Save metrics to .csv file, adding as first column the panorama ID. Add header
    with open(os.path.join(directory, f'avg_metrics_{args.threshold}_{args.radius}.csv'), 'a') as f:
        writer = csv.writer(f)
        # If the first row contains the header, don't write it again
        if os.stat(os.path.join(directory, f'avg_metrics_{args.threshold}_{args.radius}.csv')).st_size == 0:
            writer.writerow(['', 'avg_cp_distance', 'avg_ap_distance', 'avg_iou', \
                             'cp_precision', 'ap_precision', 'cp_recall', 'ap_recall', \
                            'cp_f1', 'ap_f1', 'cp_ap', 'ap_ap', 'ap50', 'ap75'])
        # Write a row with the first element as 'Average', and the others as metrics_df['column']
        print(f'Saving average metrics over {len(metrics_df)} panos to avg_metrics_{args.threshold}_{args.radius}.csv')
        writer.writerow(['Average'] + list(metrics_df.mean().round(3)))

def main(args):
    # Replace everything that is not a character with an underscore in neighbourhood string, and make it lowercase
    args.neighbourhood = re.sub(r'[^a-zA-Z]', '_', args.neighbourhood).lower()

    reoriented_foldername = 'reoriented_seg' if args.seg else 'reoriented'

    if args.ps:
        args.input_dir = os.path.join(args.input_dir, reoriented_foldername)
    else:
        args.input_dir = os.path.join(args.input_dir, args.neighbourhood)
        args.input_dir = args.input_dir + '_' + args.quality
        args.input_dir = os.path.join(args.input_dir, reoriented_foldername)

    # Create the output directory if it doesn't exist
    directory_name = f'{args.output_dir}_seg' if args.seg else f'{args.output_dir}'
    directory = os.path.join(os.path.dirname(args.input_dir), directory_name)

    if args.test:
        #args.input_dir += '_testset'
        args.backprojected_dir += '_testset'
        directory += '_testset'

    if not os.path.exists(directory):
        os.makedirs(directory)

    print(f'Input directory: {args.input_dir}')
    print(f'Backprojected directory: {args.backprojected_dir}')
    print(f'Output directory: {directory}')

    evaluate(args, directory)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='res/dataset')
    parser.add_argument('--output_dir', type=str, default='evaluation')
    parser.add_argument('--backprojected_dir', type=str, default='backprojected')
    parser.add_argument('--neighbourhood', type=str, default='osdorp')
    parser.add_argument('--label_dump', type=str, default='res/labels/labels.csv')
    parser.add_argument('--quality', type=str, default='full')
    parser.add_argument('--visualize', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--threshold', type=int, default=100, help='Threshold for distance-based metrics (in pixels)')
    parser.add_argument('--radius', type=int, default=100, help='Radius for point-to-mask best IoU (in pixels)')
    parser.add_argument('--ps', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--seg', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--test', type=bool, default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    main(args)
