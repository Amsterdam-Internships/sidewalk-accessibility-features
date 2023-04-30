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
import cv2
import matplotlib.pyplot as plt

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

    # Flatten closest_points and gt_indices, handling the case of single values
    flattened_closest_points = [
        p for sublist in closest_points for p in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])
    ]
    flattened_gt_indices = [
        i for sublist in gt_indices for i in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])
    ]

    assert len(flattened_closest_points) == len(flattened_gt_indices)

    # Remove space between rows and adjust the space between title and first row
    fig.subplots_adjust(hspace=0, top=0.9)

    for idx, (pred_mask, (row_idx, col_idx)) in enumerate(zip(pred_masks, np.ndindex((num_rows, 2)))):
        if num_rows > 1:
            ax = axes[row_idx, col_idx]
        else:
            ax = axes[col_idx]
        ax.imshow(pred_mask)  # Display the predicted mask

        # Retrieve the correct ground truth point, distance, and closest point
        gt_point = gt_points[flattened_gt_indices[idx]]
        closest_point = flattened_closest_points[idx]

        y, x = gt_point
        closest_y, closest_x = tuple(closest_point)

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


def visualize_best_dilated(args, gt_points, pred_masks, best_gt_point_indices_list, pano_id, path):
    num_masks = len(pred_masks)
    num_rows = int(np.ceil(num_masks / 2))
    fig, axes = plt.subplots(num_rows, 2, figsize=(13, 4 * num_rows))

    assert len(pred_masks) == len(best_gt_point_indices_list)

    # Remove space between rows and adjust the space between title and first row
    fig.subplots_adjust(hspace=0, top=0.9)

    # Generate a grid of coordinates for the entire image
    y_coords, x_coords = np.indices(pred_masks[0].shape)
    coords = np.column_stack((y_coords.ravel(), x_coords.ravel()))

    for idx, (pred_mask, best_gt_point_indices) in enumerate(zip(pred_masks, best_gt_point_indices_list)):
        # Determine the position of the current mask in the plot grid
        row_idx = idx // 2
        col_idx = idx % 2

        # Access the axes based on its dimensions
        if num_rows > 1:
            ax = axes[row_idx, col_idx]
        else:
            ax = axes[col_idx]

        # Ensure best_gt_point_indices is a list
        if not isinstance(best_gt_point_indices, (list, np.ndarray)):
            best_gt_point_indices = [best_gt_point_indices]
        
        label_added = False
        for best_gt_point_idx in best_gt_point_indices:
            best_gt_point = gt_points[best_gt_point_idx]

            # Compute the distance from the best ground truth point to all other points
            distances = cdist([best_gt_point], coords).reshape(pred_mask.shape)

            # Create a dilated mask by thresholding the distance matrix at the specified radius
            gt_mask_dilated = (distances <= args.radius).astype(int)

            if not label_added:
                # Display the dilated ground truth mask
                ax.imshow(gt_mask_dilated, alpha=0.5, cmap='gray', label='Dilated Ground Truth')
                ax.scatter(best_gt_point[1], best_gt_point[0], s=50, c='red', marker='x', label='Ground Truth')
                label_added = True
            else:
                ax.imshow(gt_mask_dilated, alpha=0.5, cmap='gray')
                ax.scatter(best_gt_point[1], best_gt_point[0], s=50, c='red', marker='x')

        # Display the predicted mask
        ax.imshow(pred_mask, alpha=0.5, cmap='jet', label='Predicted Mask')

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

'''def mask_to_point_distance(masks, gt_points):
    distances = []
    points = []
    gt_indices = []

    for idx, pred_mask in enumerate(masks):
        if pred_mask.ndim == 2:  # Check if the mask is 2D
            mask_channel = pred_mask
        elif pred_mask.ndim == 3:  # Check if the mask is 3D
            mask_channel = pred_mask[:, :, 0]  # Assuming the first channel contains the binary mask
        else:
            raise ValueError("Invalid mask dimensions")

        mask_coords = np.where(mask_channel > 0)  # Get the indices of non-zero elements (i.e., the mask)
        mask_coords = np.vstack(mask_coords).T  # Stack the mask indices into a 2D array

        mask_points = []

        if closest_point:
            mask_distances = []
            mask_gt_indices = []

            for gt_idx, gt_point in enumerate(gt_points):
                point_coords = np.array(gt_point).reshape(1, -1)
                dist = cdist(point_coords, mask_coords)
                local_min_distance_idx = np.argmin(dist)
                local_min_distance = dist[0, local_min_distance_idx]
                local_closest_y, local_closest_x = mask_coords[local_min_distance_idx]

                mask_distances.append(local_min_distance)
                mask_points.append((local_closest_y, local_closest_x))
                mask_gt_indices.append(gt_idx)

            distances.append(mask_distances)
            points.append(mask_points)
            gt_indices.append(mask_gt_indices)

        else:
            # Compute the anchor point as the centroid of the mask coordinates
            anchor_point = np.mean(mask_coords, axis=0)
            mask_distances = []
            mask_gt_indices = []
            anchor_points = []

            for gt_idx, gt_point in enumerate(gt_points):
                point_coords = np.array(gt_point).reshape(1, -1)
                dist = cdist(point_coords, anchor_point.reshape(1, -1))  # Calculate distance to anchor_point
                local_min_distance = dist[0, 0]

                mask_distances.append(local_min_distance)
                anchor_points.append(tuple(anchor_point))
                mask_gt_indices.append(gt_idx)


            distances.append(mask_distances)
            points.append(anchor_points)
            gt_indices.append(mask_gt_indices)

    return distances, points, gt_indices'''

def mask_to_point_distance(pred_masks, gt_points, use_centroid=True):
    closest_distances = []
    anchor_distances = []
    closest_points = []
    anchor_points = []
    mask_indices = []
    gt_indices = []

    for gt_idx, gt_point in enumerate(gt_points):
        point_coords = np.array(gt_point).reshape(1, -1)

        min_closest_distance = float('inf')
        min_anchor_distance = float('inf')
        closest_y, closest_x = None, None
        best_anchor_point = None
        best_mask_idx = None

        for mask_idx, mask_dict in enumerate(pred_masks):
            rle = mask_dict['segmentation']
            mask_channel = mask_util.decode(rle)

            mask_coords = np.where(mask_channel > 0)  # Get the indices of non-zero elements (i.e., the mask)
            mask_coords = np.vstack(mask_coords).T  # Stack the mask indices into a 2D array

            # Closest point distance
            dist = cdist(point_coords, mask_coords)
            local_min_distance_idx = np.argmin(dist)
            local_min_distance = dist[0, local_min_distance_idx]
            local_closest_y, local_closest_x = mask_coords[local_min_distance_idx]

            # Anchor point distance
            anchor_point = mask_dict['centroid']
            anchor_dist = cdist(point_coords, np.array(anchor_point).reshape(1, -1))  # Calculate distance to anchor_point
            local_anchor_distance = anchor_dist[0, 0]

            if use_centroid:
                if local_anchor_distance < min_anchor_distance:
                    min_anchor_distance = local_anchor_distance
                    best_anchor_point = anchor_point
                    best_mask_idx = mask_idx
            else:
                if local_min_distance < min_closest_distance:
                    min_closest_distance = local_min_distance
                    closest_y, closest_x = local_closest_y, local_closest_x
                    best_mask_idx = mask_idx

        if use_centroid:
            anchor_distances.append(min_anchor_distance)
            anchor_points.append(best_anchor_point)
        else:
            closest_distances.append(min_closest_distance)
            closest_points.append((closest_y, closest_x))

        mask_indices.append(best_mask_idx)
        print(f'Area of the best mask {best_mask_idx}: {pred_masks[best_mask_idx]["area"]}')
        gt_indices.append(gt_idx)

    # Test: visualize the best masks
    # Open image
    path = '/var/scratch/lombardo/download_PS/reprojected/9VW3AUbKBdSPqnURnwRSLQ/left.png'
    img = cv2.imread(path)
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    # Draw the best masks
    for mask_idx, mask_dict in enumerate(pred_masks):
        if mask_idx in mask_indices:
            print(f'Mask {mask_idx}')
            rle = mask_dict['segmentation']
            mask_channel = mask_util.decode(rle)
            plt.imshow(mask_channel, alpha=0.5)
            # Visualize ground truth point
            plt.scatter(gt_points[mask_indices.index(mask_idx)][1], gt_points[mask_indices.index(mask_idx)][0], c='r', s=7)
            # Visualize closest point
            if not use_centroid:
                plt.scatter(closest_points[mask_indices.index(mask_idx)][1], closest_points[mask_indices.index(mask_idx)][0], c='g', s=7)
            # Visualize closest anchor point
            else:
                plt.scatter(anchor_points[mask_indices.index(mask_idx)][1], anchor_points[mask_indices.index(mask_idx)][0], c='blue', s=7)
    plt.savefig('test.png')
    plt.close()



    if use_centroid:
        # Round the distances to the third decimal place

        return np.round(anchor_distances, 3), anchor_points, mask_indices, gt_indices
    else:
        return np.round(closest_distances, 3), closest_points, mask_indices, gt_indices

def mask_to_point_best_iou(gt_points, pred_masks, radius=5):
    best_ious = []
    best_gt_point_indices = []

    # Generate a grid of coordinates for the entire image
    y_coords, x_coords = np.indices(pred_masks[0].shape)
    coords = np.column_stack((y_coords.ravel(), x_coords.ravel()))

    for idx, pred_mask in enumerate(pred_masks):
        mask_ious = []
        mask_gt_point_indices = []

        # Get a random point in the mask and find its image part
        mask_points = np.where(pred_mask)
        random_index = random.randint(0, len(mask_points[0]) - 1)
        random_mask_point = (mask_points[0][random_index], mask_points[1][random_index])
        mask_part = get_image_part(random_mask_point[1])

        for point_idx, gt_point in enumerate(gt_points):
            gt_part = get_image_part(gt_point[1])

            # Check if the mask and the label are both in the same part of the image
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

                mask_ious.append(iou)
                mask_gt_point_indices.append(point_idx)

        # If there are no points in the same part, append a large distance and a dummy point
        if len(mask_ious) > 0:
            best_ious.append(mask_ious)
            best_gt_point_indices.append(mask_gt_point_indices)
        else:
            best_ious.append([-1])
            best_gt_point_indices.append([-1])

    return best_ious, best_gt_point_indices

def precision_recall_f1(distances, gt_indices, threshold):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    assert len(distances) == len(gt_indices)

    matched_gt_indices = set()

    # Count true positives and false positives
    for dist_list, idx_list in zip(distances, gt_indices):
        # Handle the case where there is only one valid ground truth point
        if not isinstance(dist_list, (list, np.ndarray)):
            dist_list = [dist_list]
        if not isinstance(idx_list, (list, np.ndarray)):
            idx_list = [idx_list]

    for distance, gt_idx in zip(dist_list, idx_list):
        for distance, gt_idx in zip(dist_list, idx_list):
            if distance <= threshold:
                true_positives += 1
                matched_gt_indices.add(gt_idx)
            else:
                false_positives += 1

    # Count false negatives
    all_gt_indices = {i for idx_list in gt_indices for i in (idx_list if isinstance(idx_list, (list, np.ndarray)) else [idx_list])}
    false_negatives = len(all_gt_indices) - len(matched_gt_indices)

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

    # print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1_score:.3f}")
    return precision, recall, f1_score

def average_precision(distances, gt_indices, threshold):
    true_positives = []
    false_positives = []

    all_gt_indices = {i for idx_list in gt_indices for i in (idx_list if isinstance(idx_list, (list, np.ndarray)) else [idx_list])}
    n_gt_points = len(all_gt_indices)

    assert len(distances) == len(gt_indices)

    # Count true positives and false positives
    for dist_list, idx_list in zip(distances, gt_indices):
        # Handle the case where there is only one valid ground truth point
        if not isinstance(dist_list, (list, np.ndarray)):
            dist_list = [dist_list]
        if not isinstance(idx_list, (list, np.ndarray)):
            idx_list = [idx_list]
        for distance, gt_idx in zip(dist_list, idx_list):
            if distance <= threshold:
                true_positives.append(1)
                false_positives.append(0)
            else:
                true_positives.append(0)
                false_positives.append(1)

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
    all_best_gt_indices = {i for idx_list in best_gt_point_indices for i in (idx_list if isinstance(idx_list, (list, np.ndarray)) else [idx_list])}
    num_gt_points = len(all_best_gt_indices)

    flattened_best_ious = [iou for iou_list in best_ious for iou in (iou_list if isinstance(iou_list, (list, np.ndarray)) else [iou_list])]

    num_pred_masks = len(flattened_best_ious)

    sorted_indices = sorted(range(num_pred_masks), key=lambda i: flattened_best_ious[i], reverse=True)
    sorted_best_ious = [flattened_best_ious[i] for i in sorted_indices]

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

def evaluate_single_face(masks, labels_df, directory):
    '''Access the list of dictionaries masks for the current pano and face. The structure is the following:
    masks = [
                {
                'pano_id': 'pano_id_0',
                'face_idx': 'face_idx_0',
                'area': 'area',
                'centroid': [x, y],
                'segmentation': {
                    'size': [height, width],
                    'counts': 'RLE'
                    },
                },
                {...}, {...}, ...
            ]
    '''
    # Traverse masks list, check ['segmentation']['counts'], decode it and check if it's the same as the next one in the list
    for i in range(len(masks)):
        mask = masks[i]
        # Decode RLE
        mask_decoded = mask_util.decode(mask['segmentation'])
        # Check if the current mask is the same as the next one
        if i < len(masks) - 1:
            next_mask = masks[i + 1]
            next_mask_decoded = mask_util.decode(next_mask['segmentation'])
            if np.array_equal(mask_decoded, next_mask_decoded):
                print(f"Mask {i} and mask {i + 1} are the same")
                # Check if their areas are also the same
                if mask['area'] == next_mask['area']:
                    print(f"Mask {i} and mask {i + 1} have the same area")
                else:
                    print(f"BUT...Mask {i} and mask {i + 1} have different areas")
            else:
                print(f"Mask {i} and mask {i + 1} are different")
    pano_id = masks[0]['pano_id']
    face_idx = masks[0]['face_idx']
    print(pano_id)
    return
    # Load ground truth points knowing that the coordinates are (reprojected_pano_y,reprojected_pano_x)
    gt_points = []
    for idx, row in labels_df.iterrows():
            if row['gsv_panorama_id'] == pano_id and row['face_idx'] == face_idx:
                gt_points.append([row['reprojected_pano_y'], row['reprojected_pano_x']])
    print(f"Number of ground truth points: {len(gt_points)}")
    print(f"Ground truth points: {gt_points}")
    
    # Compute metrics
    # Metrics 1: (average) mask-to-point distance
    closest_distances, closest_points, mask_indices, gt_indices = mask_to_point_distance(masks, gt_points, True)
    print(f"Closest distances: {closest_distances}, Closest points: {closest_points}, Closest masks: {mask_indices}," \
            f"GT indices related to the closest masks: {gt_indices}")
    #anchor_distances, anchor_points, mask_indices, gt_indices = mask_to_point_distance(masks, gt_points, True)
    #print(f"Anchor distances: {anchor_distances}, Anchor points: {anchor_points}, Anchor masks: {mask_indices}," \
    #        f"GT indices related to the anchor masks: {gt_indices}")

    '''# Compute metrics
    # Metrics 1: (average) mask-to-closest-point distance
    cp_distances, cp_points, cp_gt_indices, is_empty = mask_to_point_distance(gt_points, pred_masks, True)
    if is_empty:
        continue
    # Calculate mean distance between all masks and points, closest points only
    # Exclude float('inf') values
    cp_distances_filtered = [d for dist_list in cp_distances for d in dist_list if not np.isinf(d)]
    cp_mean_distance = np.mean(cp_distances_filtered)

    # Metrics 1.1: (average) mask-to-closest-anchor-point distance
    ap_distances, ap_points, ap_gt_indices, is_empty = mask_to_point_distance(gt_points, pred_masks, False)
    ap_distances_filtered = [d for dist_list in ap_distances for d in dist_list if not np.isinf(d)]
    ap_mean_distance = np.mean(ap_distances_filtered)

    # Metrics 2: point-to-mask best IoU
    # Exclude -1 values
    best_ious, best_gt_point_indices = mask_to_point_best_iou(gt_points, pred_masks, radius=100)
    best_ious_filtered = [iou for iou_list in best_ious for iou in iou_list if iou != -1]
    mean_ious = np.mean(best_ious_filtered)

    # Filter valid quadruples based on gt_indices
    valid_cp_quadruples = [(m, d, p, i) for m, dist_list, point_list, idx_list in zip(pred_masks, cp_distances, cp_points, cp_gt_indices)
            for d, p, i in zip(
                dist_list if isinstance(dist_list, (list, np.ndarray)) else [dist_list],
                point_list if isinstance(point_list, (list, np.ndarray)) else [point_list],
                idx_list if isinstance(idx_list, (list, np.ndarray)) else [idx_list]
            ) if i != -1]

    valid_ap_quadruples = [(m, d, p, i) for m, dist_list, point_list, idx_list in zip(pred_masks, ap_distances, ap_points, ap_gt_indices)
                        for d, p, i in zip(
                            dist_list if isinstance(dist_list, (list, np.ndarray)) else [dist_list],
                            point_list if isinstance(point_list, (list, np.ndarray)) else [point_list],
                            idx_list if isinstance(idx_list, (list, np.ndarray)) else [idx_list]
                        ) if i != -1]

    valid_iou_triples = [(m, iou, i) for m, iou_list, idx_list in zip(pred_masks, best_ious, best_gt_point_indices)
                        for iou, i in zip(
                            iou_list if isinstance(iou_list, (list, np.ndarray)) else [iou_list],
                            idx_list if isinstance(idx_list, (list, np.ndarray)) else [idx_list]
                        ) if i != -1]

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
                                    pano, pano_path) # metrics 2'''

def evaluate(args, directory):

    # Load labels from .csv file
    labels_df = pd.read_csv(args.labels_csv_path)

    # Count number of labels (rows)
    num_labels = len(labels_df.index)
    print(f'Number of labels: {num_labels}')

    # Load path to panos
    pano_path = args.input_dir

    # Load masks from .json file
    '''The panos_info dictionary will have the following structure:
        panos_info = [
                {
                'pano_id': {
                    'face_idx': {
                        'masks': [...],
                        'areas': [...],
                        'centroids': [...]
                    },
                    ...
                },
                ...
            }
        ]'''
    json_path = os.path.join(args.masks_dir, 'input_coco_format.json')
    with open(json_path) as f:
        panos_info = json.load(f)

    # Count number of unique panos (pano is a dictionary)
    pano_ids = set(pano['pano_id'] for pano in panos_info)
    num_panoramas = len(pano_ids)
    print(f'Number of panos: {num_panoramas}')
    # Count number of masks (panos_info['pano_id']['face_idx']['masks'])
    print(f'Number of masks: {len(panos_info)}')

    '''The structure of grouped_panos will be:
    grouped_panos = {
        'pano_id_0': {
            'face_idx_0': [
                {
                    'pano_id': 'pano_id_0',
                    'face_idx': 'face_idx_0',
                    'area': 'area',
                    'centroid': [x, y],
                    'segmentation': {
                        'size': [height, width],
                        'counts': 'RLE'
                        },
                },
                {...},
                ...
            ]
            'face_idx_1': [...],
            ...
        },
        'pano_id_1': {}
        ...
    }'''
    grouped_panos = {}
    for pano in panos_info:
        pano_id = pano['pano_id']
        face_idx = pano['face_idx']

        if pano_id not in grouped_panos:
            grouped_panos[pano_id] = {}

        if face_idx not in grouped_panos[pano_id]:
            grouped_panos[pano_id][face_idx] = []

        grouped_panos[pano_id][face_idx].append(pano)

    # Evaluate each panorama's masks
    for pano_id in tqdm(grouped_panos):
        print(f'Processing pano {pano_id}:')
        for face_idx in grouped_panos[pano_id]:
            # Check that the face has masks
            if len(grouped_panos[pano_id][face_idx]) == 0:
                print(f'Face {face_idx} has no masks. Skipping...')
                continue

            # Check if the face has labels
            if not labels_df[(labels_df['gsv_panorama_id'] == pano_id) & (labels_df['face_idx'] == face_idx)].empty:
                print(f'Number of masks for face {face_idx}: {len(grouped_panos[pano_id][face_idx])}')
                n_labels = len(labels_df[(labels_df["gsv_panorama_id"] == pano_id) & (labels_df["face_idx"] == face_idx)])
                print(f'Number of labels for face {face_idx}: {n_labels}')
                masks = grouped_panos[pano_id][face_idx]
                evaluate_single_face(masks, labels_df, directory)
                break
            else:
                print(f'Face {face_idx} has no labels. Skipping...')
        break
    
    # Calculate average metrics
    '''# Open the .csv file, make it a dataframe, calculate the average of each column,
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
        writer.writerow(['Average'] + list(metrics_df.mean().round(3)))'''

def main(args):

    # Set a seed to reproduce the same randomness of panos visualization
    random.seed(42)
    
    # Create directory to save results
    directory = args.output_dir

    if not os.path.exists(directory):
        os.makedirs(directory)

    print(f'Input directory: {args.input_dir}')
    print(f'Masks directory: {args.masks_dir}')
    print(f'Labels csv path: {args.labels_csv_path}')
    print(f'Threshold: {args.threshold}')
    print(f'Radius: {args.radius}')
    print(f'Output directory: {directory}')

    evaluate(args, directory)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='res/dataset')
    parser.add_argument('--masks_dir', type=str, default='masks')
    parser.add_argument('--labels_csv_path', type=str, default='res/labels/labels.csv')
    parser.add_argument('--threshold', type=int, default=100, help='Threshold for distance-based metrics (in pixels)')
    parser.add_argument('--radius', type=int, default=100, help='Radius for point-to-mask best IoU (in pixels)')
    parser.add_argument('--output_dir', type=str, default='evaluation')

    args = parser.parse_args()

    main(args)