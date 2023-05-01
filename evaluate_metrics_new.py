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
import concurrent.futures
import psutil
import pickle

def map_faces(index):
    # Function that maps the face index to the corresponding face
    # Mapping: 0F 1R 2B 3L 4U 5D
    # For example: index == 0, return 'front'
    if index == 0:
        return 'front'
    elif index == 1:
        return 'right'
    elif index == 2:
        return 'back'
    elif index == 3:
        return 'left'
    elif index == 4:
        return 'top'
    elif index == 5:
        return 'bottom'

def visualize_labels(args, faces, pano_id, masks, labels_df, directory):

    for face_idx in faces:
        # Reverse map the face index to the face name
        face = map_faces(int(face_idx))

        # Check if we already processed this face
        output_path = os.path.join(directory, f'visualize_{pano_id}_{face}.png')
        if os.path.exists(output_path):
            continue

        fig = plt.figure()
        # Load the face
        face_path = os.path.join(args.input_dir, pano_id, f'{face}.png')
        img = cv2.imread(face_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)

        # Traverse the masks in masks_list, decode them and load each of them with a different color
        output_mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        for masks_face_idx, masks_list in masks.items():
            if masks_face_idx == int(face_idx):
                for idx, mask_dict in enumerate(masks_list):
                    rle = mask_dict['segmentation']
                    mask = mask_util.decode(rle)
                    color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                    output_mask[mask == 1] = color

        # Plot the mask
        plt.imshow(output_mask, alpha=0.4)
        #mask_path = os.path.join(args.masks_dir, pano_id, f'{face}.png')
        #mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        #plt.imshow(mask, alpha=0.5, cmap='gray')

        # Load ground truth points knowing that the coordinates are (reprojected_pano_y,reprojected_pano_x)
        gt_points = []
        for idx, row in labels_df.iterrows():
                if row['gsv_panorama_id'] == pano_id.replace("'", "") and row['face_idx'] == int(face_idx):
                    gt_points.append([row['reprojected_pano_y'], row['reprojected_pano_x']])

        for gt_point in gt_points:
            plt.scatter(gt_point[1], gt_point[0], color='red', s=7)

        # Set small tick labels
        plt.tick_params(axis='both', which='both', labelsize=6)

        # Add legend for red points
        plt.scatter([], [], color='red', s=8, label='Ground Truth')
        plt.legend(prop={'size': 5})

        # Save the image
        fig.savefig(os.path.join(directory, f'visualize_{pano_id}_{face}.png'), dpi=300, bbox_inches='tight')
        # Close the figure
        plt.close(fig)

def visualize_closest_mask(args, faces, pano_id, masks, labels_df, directory, use_centroid=True):
    
    for face_idx in faces:
        # Reverse map the face index to the face name
        face = map_faces(int(face_idx))

        # Check if we already processed this face
        centroid = 'ap' if use_centroid else 'cp'
        output_path = os.path.join(directory, f'{centroid}_{pano_id}_{face}.png')
        if os.path.exists(output_path):
            continue

        fig = plt.figure()
        # Load ground truth points knowing that the coordinates are (reprojected_pano_y,reprojected_pano_x)
        gt_points = []
        for idx, row in labels_df.iterrows():
                if row['gsv_panorama_id'] == pano_id.replace("'", "") and row['face_idx'] == int(face_idx):
                    gt_points.append([row['reprojected_pano_y'], row['reprojected_pano_x']])

        # The mask index is in either "closest_points" or "anchor_points" depending on use_centroid
        key = 'anchor_points' if use_centroid else 'closest_points'
        for idx, (mask_idx, gt_idx) in enumerate(zip(faces[face_idx][key]['indices']['mask_indices'], faces[face_idx][key]['indices']['gt_indices'])):
            # mask_idx is the index of the dictionary in masks where the mask is stored
            # Convert the dictionary values to a list
            for masks_face_idx, masks_list in masks.items():
                # TODO: Remove after next evaluation (we fix -1 error)
                if masks_face_idx == -1:
                    masks_face_idx == 0
                if masks_face_idx == int(face_idx):
                    for m_idx, mask_dict in enumerate(masks_list):
                        if m_idx == mask_idx:
                            rle = mask_dict['segmentation']
                            mask = mask_util.decode(rle)
                            plt.imshow(mask)
                            break

            # Get the closest point
            mask_point = faces[face_idx][key]['points'][idx]
            point_color = 'blue' if use_centroid else 'green'
            plt.scatter(mask_point[1], mask_point[0], color=point_color, s=7)
            # Get the ground truth point
            gt_point = gt_points[gt_idx]
            plt.scatter(gt_point[1], gt_point[0], color='red', s=7)
            # Draw a dashed line between the two points
            plt.plot([mask_point[1], gt_point[1]], [mask_point[0], gt_point[0]], color='black', linestyle='dashed', linewidth=1)

        # Set small tick labels
        plt.tick_params(axis='both', which='both', labelsize=6)

        # Add legend for red points
        plt.scatter([], [], color='red', s=8, label='Ground Truth')
        label = 'Anchor Point' if use_centroid else 'Closest Point'
        plt.scatter([], [], color=point_color, s=8, label=label)
        plt.legend(prop={'size': 5})

        # Save the image
        fig.savefig(os.path.join(directory, output_path), dpi=300, bbox_inches='tight')
        # Close the figure
        plt.close(fig)

def visualize_best_iou_mask(args, faces, pano_id, masks, labels_df, directory):
    
    for face_idx in faces:
        # Reverse map the face index to the face name
        face = map_faces(int(face_idx))

        # Check if we already processed this face
        output_path = os.path.join(directory, f'iou_{pano_id}_{face}.png')
        if os.path.exists(output_path):
            continue

        fig = plt.figure()
        # Load ground truth points knowing that the coordinates are (reprojected_pano_y,reprojected_pano_x)
        gt_points = []
        for idx, row in labels_df.iterrows():
                if row['gsv_panorama_id'] == pano_id.replace("'", "") and row['face_idx'] == int(face_idx):
                    gt_points.append([row['reprojected_pano_y'], row['reprojected_pano_x']])

        # The mask index is in 'iou_indices'
        first_iteration = True
        for idx, (mask_idx, gt_idx) in enumerate(zip(faces[face_idx]['iou_indices']['mask_indices'], faces[face_idx]['iou_indices']['gt_indices'])):
            # mask_idx is the index of the dictionary in masks where the mask is stored
            # Convert the dictionary values to a list
            for masks_face_idx, masks_list in masks.items():
                if masks_face_idx == int(face_idx):
                    for m_idx, mask_dict in enumerate(masks_list):
                        if m_idx == mask_idx:
                            rle = mask_dict['segmentation']
                            mask = mask_util.decode(rle)
                            if first_iteration:
                                combined_gt_mask_dilated = np.zeros_like(mask, dtype=int)
                                first_iteration = False
                            plt.imshow(mask)
                            break
            # Get the ground truth point
            gt_point = gt_points[gt_idx]
            plt.scatter(gt_point[1], gt_point[0], color='red', s=7)

            # Generate a grid of coordinates for the entire image
            y_coords, x_coords = np.indices(mask.shape)
            coords = np.column_stack((y_coords.ravel(), x_coords.ravel()))

            # Visualize dilated mask of the ground truth point
            # Compute the distance from the best ground truth point to all other points
            distances = cdist([gt_point], coords).reshape(mask.shape)

            # Create a dilated mask by thresholding the distance matrix at the specified radius
            gt_mask_dilated = (distances <= args.radius).astype(int)

            # Accumulate the dilated ground truth masks
            combined_gt_mask_dilated |= gt_mask_dilated

        # Visualize the combined_gt_mask_dilated
        plt.imshow(combined_gt_mask_dilated, alpha=0.5, cmap='gray')

        # Set small tick labels
        plt.tick_params(axis='both', which='both', labelsize=6)

        # Add legend for red points
        plt.scatter([], [], color='red', s=7, label='Ground Truth')
        plt.scatter([], [], color='white', s=10, label='Dilated Ground Truth')
        plt.legend(prop={'size': 5})

        # Save the image
        fig.savefig(os.path.join(directory, output_path), dpi=300, bbox_inches='tight')
        # Close the figure
        plt.close(fig)


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
            anchor_point = mask_dict['centroid'][::-1]
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
        gt_indices.append(gt_idx)

    '''# Test: visualize the best masks
    # Open image
    pano_id = pred_masks[0]['pano_id']
    path = f'/var/scratch/lombardo/download_PS/reprojected/{pano_id}/right.png'
    img = cv2.imread(path)
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    mask_path = f'/var/scratch/lombardo/download_PS/masks_resized/{pano_id}/right.png'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #plt.imshow(mask, alpha=0.3)
    # Draw the best masks
    for mask_idx, mask_dict in enumerate(pred_masks):
        if mask_idx in mask_indices:
            print(f'Mask {mask_idx}')
            rle = mask_dict['segmentation']
            mask_channel = mask_util.decode(rle)
            plt.imshow(mask_channel, alpha=0.3)
            # Visualize ground truth point
            plt.scatter(gt_points[mask_indices.index(mask_idx)][1], gt_points[mask_indices.index(mask_idx)][0], c='r', s=7)
            # Visualize closest point
            if not use_centroid:
                plt.scatter(closest_points[mask_indices.index(mask_idx)][1], closest_points[mask_indices.index(mask_idx)][0], c='g', s=7)
            # Visualize closest anchor point
            else:
                plt.scatter(anchor_points[mask_indices.index(mask_idx)][1], anchor_points[mask_indices.index(mask_idx)][0], c='blue', s=7)
    plt.savefig('test.png')
    plt.close()'''



    if use_centroid:
        # Round the distances to the third decimal place
        return np.round(anchor_distances, 3), np.round(anchor_points, 3), mask_indices, gt_indices
    else:
        return np.round(closest_distances, 3), np.round(closest_points, 3), mask_indices, gt_indices

def mask_to_point_best_iou(pred_masks, gt_points, radius=5):
    best_ious = []
    best_mask_indices = []
    gt_indices = []

    # Generate a grid of coordinates for the entire image
    rle = pred_masks[0]['segmentation']
    mask_shape = mask_util.decode(rle).shape
    y_coords, x_coords = np.indices(mask_shape)
    coords = np.column_stack((y_coords.ravel(), x_coords.ravel()))

    for point_idx, gt_point in enumerate(gt_points):
        point_coords = np.array(gt_point).reshape(1, -1)

        max_iou = -1
        best_mask_idx = -1

        for mask_idx, mask_dict in enumerate(pred_masks):
            rle = mask_dict['segmentation']
            pred_mask = mask_util.decode(rle)

            # Compute the distance from the ground truth point to all other points
            distances = cdist(point_coords, coords).reshape(pred_mask.shape)
            # Create a dilated mask by thresholding the distance matrix at the specified radius
            gt_mask_dilated = (distances <= radius).astype(int)

            # Calculate IoU
            intersection = np.sum(np.logical_and(gt_mask_dilated, pred_mask))
            union = np.sum(np.logical_or(gt_mask_dilated, pred_mask))
            iou = intersection / union

            if iou > max_iou:
                max_iou = iou
                best_mask_idx = mask_idx

        best_ious.append(max_iou)
        best_mask_indices.append(best_mask_idx)
        gt_indices.append(point_idx)

    '''if pred_masks[0]['pano_id'] == 'AcsKZFPqjaAAH8BkvTmtPA':
        # Test visualization
        # Test: visualize the best masks
        # Open image
        pano_id = pred_masks[0]['pano_id']
        face = 'back'
        path = f'/var/scratch/lombardo/download_PS/reprojected/{pano_id}/{face}.png'
        img = cv2.imread(path)
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        mask_path = f'/var/scratch/lombardo/download_PS/masks_resized/{pano_id}/{face}.png'
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        plt.imshow(mask, alpha=0.3)
        # Visualize ground truth points
        for gt_point in gt_points:
            plt.scatter(gt_point[1], gt_point[0], c='r', s=7)
        # Draw the best masks
        for mask_idx, mask_dict in enumerate(pred_masks):
            if mask_idx in best_mask_indices:
                print(f'Mask {mask_idx}')
                rle = mask_dict['segmentation']
                mask_channel = mask_util.decode(rle)
                #plt.imshow(mask_channel, alpha=0.3)
                # Visualize ground truth point
                #plt.scatter(gt_points[best_mask_indices.index(mask_idx)][1], gt_points[best_mask_indices.index(mask_idx)][0], c='r', s=7)

                # Visualize dilated mask of the ground truth point
                # Compute the distance from the best ground truth point to all other points
                distances = cdist([gt_points[best_mask_indices.index(mask_idx)]], coords).reshape(mask_channel.shape)

                # Create a dilated mask by thresholding the distance matrix at the specified radius
                gt_mask_dilated = (distances <= args.radius).astype(int)

                # Display the dilated ground truth mask
                plt.imshow(gt_mask_dilated, alpha=0.5, cmap='gray', label='Dilated Ground Truth')
                
        plt.savefig('test.png')
        plt.close()'''
    return best_ious, best_mask_indices, gt_indices

def precision_recall_f1(distances, gt_indices, threshold):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    assert len(distances) == len(gt_indices)

    matched_gt_indices = set()

    for dist_list, idx_list in zip(distances, gt_indices):
        # Handle the case where there is only one valid ground truth point
        if not isinstance(dist_list, (list, np.ndarray)):
            dist_list = [dist_list]
        if not isinstance(idx_list, (list, np.ndarray)):
            idx_list = [idx_list]

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
    pano_id = masks[0]['pano_id']
    face_idx = masks[0]['face_idx']
    print(pano_id)
    # Load ground truth points knowing that the coordinates are (reprojected_pano_y,reprojected_pano_x)
    gt_points = []
    for idx, row in labels_df.iterrows():
            if row['gsv_panorama_id'] == pano_id and row['face_idx'] == face_idx:
                gt_points.append([row['reprojected_pano_y'], row['reprojected_pano_x']])
    print(f"Number of ground truth points: {len(gt_points)}")
    print(f"Ground truth points: {gt_points}")
    
    # Compute metrics
    # Metrics 1: mask-to-point distance
    closest_distances, closest_points, cp_mask_indices, cp_gt_indices = mask_to_point_distance(masks, gt_points, False)
    anchor_distances, anchor_points, ap_mask_indices, ap_gt_indices = mask_to_point_distance(masks, gt_points, True)
    # Metrics 2: mask-to-(dilated) mask (IoU)
    best_ious, mask_indices_iou, gt_indices_iou = mask_to_point_best_iou(masks, gt_points, args.radius)
    # Metrics 3: Precision, Recall, F1
    cp_precision, cp_recall, cp_f1 = precision_recall_f1(closest_distances, cp_gt_indices, args.threshold)
    ap_precision, ap_recall, ap_f1 = precision_recall_f1(anchor_distances, ap_gt_indices, args.threshold)
    # Metrics 4: Average Precision
    cp_ap = average_precision(closest_distances, cp_gt_indices, args.threshold)
    ap_ap = average_precision(anchor_distances, ap_gt_indices, args.threshold)
    # Metrics 5: Average precision @50 and @75 based on IoU
    ap50 = average_precision_iou(best_ious, gt_indices_iou, threshold=0.5)
    ap75 = average_precision_iou(best_ious, gt_indices_iou, threshold=0.75)

    '''Structure of the output dictionary:
        results = {
                    'pano_id_0': {
                        'face_idx_0': {
                            'closest_points': {
                                'metrics': {
                                    'distances': [],
                                    'precision': [],
                                    'recall': [],
                                    'f1': [],
                                    'average_precision': [],
                                },
                                'points': [],
                                'indices': {
                                    'mask_indices': [],
                                    'gt_indices': []
                                }
                            },
                            'anchor_points': {
                                'metrics': {
                                    'distances': [],
                                    'precision': [],
                                    'recall': [],
                                    'f1': [],
                                    'average_precision': [],
                                },
                                'points': [],
                                'indices': {
                                    'mask_indices': [],
                                    'gt_indices': []
                                }
                            },
                            'iou_metrics': {
                                'best_ious': [],
                                'ap_50': [],
                                'ap_75': []
                            },
                            'iou_indices': {
                                'mask_indices': [],
                                'gt_indices': []
                            }
                        },
                        'face_idx_1': {...},
                    },
                    'pano_id_1': {...}
                }'''
    # Return the metrics, points, and indices
    return {
        'closest_points': {
            'metrics': {
                'distances': closest_distances,
                'precision': cp_precision,
                'recall': cp_recall,
                'f1': cp_f1,
                'average_precision': cp_ap,
            },
            'points': closest_points,
            'indices': {
                'mask_indices': cp_mask_indices,
                'gt_indices': cp_gt_indices
            }
        },
        'anchor_points': {
            'metrics': {
                'distances': anchor_distances,
                'precision': ap_precision,
                'recall': ap_recall,
                'f1': ap_f1,
                'average_precision': ap_ap,
            },
            'points': anchor_points,
            'indices': {
                'mask_indices': ap_mask_indices,
                'gt_indices': ap_gt_indices
            }
        },
        'iou_metrics': {
            'best_ious': best_ious,
            'ap_50': ap50,
            'ap_75': ap75
        },
        'iou_indices': {
            'mask_indices': mask_indices_iou,
            'gt_indices': gt_indices_iou
        }
    }

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
    json_path = os.path.join(args.masks_dir, args.input_coco_format)
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
    # Save results in a dictionary
    results = {} # for each face
    mean_results = {} # for each pano

    for pano_id in tqdm(grouped_panos):
        print(f'Processing pano {pano_id}:')

        if pano_id not in results:
            results[pano_id] = {}
        
        for face_idx in grouped_panos[pano_id]:
            # Check that the face has masks
            if len(grouped_panos[pano_id][face_idx]) == 0:
                print(f'Face {face_idx} has no masks. Skipping...')
                continue
            # Check if the face has labels
            if labels_df[(labels_df['gsv_panorama_id'] == pano_id) & (labels_df['face_idx'] == face_idx)].empty:
                print(f'Face {face_idx} has no labels. Skipping...')
                continue
            else:
                print(f'Number of masks for face {face_idx}: {len(grouped_panos[pano_id][face_idx])}')
                n_labels = len(labels_df[(labels_df["gsv_panorama_id"] == pano_id) & (labels_df["face_idx"] == face_idx)])
                print(f'Number of labels for face {face_idx}: {n_labels}')
                masks = grouped_panos[pano_id][face_idx]

                if face_idx not in results[pano_id]:
                    results[pano_id][face_idx] = {}

                results[pano_id][face_idx] = evaluate_single_face(masks, labels_df, directory)

        # Initialize accumulators for each metric across all faces
        cp_total_distance = []
        ap_total_distance = []
        total_best_iou = []
        cp_total_precision = []
        cp_total_recall = []
        cp_total_f1 = []
        ap_total_precision = []
        ap_total_recall = []
        ap_total_f1 = []
        cp_total_ap = []
        ap_total_ap = []
        total_ap50 = []
        total_ap75 = []

        # Iterate through the results dictionary
        for pano_id, faces in results.items():
            if pano_id not in mean_results:
                mean_results[pano_id] = {}
            
            for face_idx, metrics in faces.items():
                # Accumulate metrics for each face
                cp_total_distance.extend(metrics['closest_points']['metrics']['distances'])
                ap_total_distance.extend(metrics['anchor_points']['metrics']['distances'])
                total_best_iou.extend(metrics['iou_metrics']['best_ious'])
                cp_total_precision.append(metrics['closest_points']['metrics']['precision'])
                cp_total_recall.append(metrics['closest_points']['metrics']['recall'])
                cp_total_f1.append(metrics['closest_points']['metrics']['f1'])
                ap_total_precision.append(metrics['anchor_points']['metrics']['precision'])
                ap_total_recall.append(metrics['anchor_points']['metrics']['recall'])
                ap_total_f1.append(metrics['anchor_points']['metrics']['f1'])
                cp_total_ap.append(metrics['closest_points']['metrics']['average_precision'])
                ap_total_ap.append(metrics['anchor_points']['metrics']['average_precision'])
                total_ap50.append(metrics['iou_metrics']['ap_50'])
                total_ap75.append(metrics['iou_metrics']['ap_75'])

        # Calculate the mean for each metric across all faces
        mean_cp_distance = np.mean(cp_total_distance)
        mean_ap_distance = np.mean(ap_total_distance)
        mean_best_iou = np.mean(total_best_iou)
        mean_cp_precision = np.mean(cp_total_precision)
        mean_cp_recall = np.mean(cp_total_recall)
        mean_cp_f1 = np.mean(cp_total_f1)
        mean_ap_precision = np.mean(ap_total_precision)
        mean_ap_recall = np.mean(ap_total_recall)
        mean_ap_f1 = np.mean(ap_total_f1)
        mean_cp_ap = np.mean(cp_total_ap)
        mean_ap_ap = np.mean(ap_total_ap)
        mean_ap50 = np.mean(total_ap50)
        mean_ap75 = np.mean(total_ap75)

        # Store the mean metrics in the mean_results dictionary
        mean_results[pano_id] = {
            'mean_cp_distance': mean_cp_distance,
            'mean_ap_distance': mean_ap_distance,
            'mean_best_iou': mean_best_iou,
            'mean_cp_precision': mean_cp_precision,
            'mean_cp_recall': mean_cp_recall,
            'mean_cp_f1': mean_cp_f1,
            'mean_ap_precision': mean_ap_precision,
            'mean_ap_recall': mean_ap_recall,
            'mean_ap_f1': mean_ap_f1,
            'mean_cp_ap': mean_cp_ap,
            'mean_ap_ap': mean_ap_ap,
            'mean_ap50': mean_ap50,
            'mean_ap75': mean_ap75
        }

        # Set the filename for the CSV file
        csv_path = os.path.join(directory, "mean_results.csv")

        # Prepare the header for the CSV file
        header = ['pano_id'] + list(mean_results[pano_id].keys())

        # Check if the file exists
        if not os.path.exists(csv_path):
            # If the file doesn't exist, create it and write the header
            with open(csv_path, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)

        
        # Prepare the row to be written in the CSV file
        row = [pano_id] + list(mean_results[pano_id].values())

        # Append the new pano results to the CSV file
        with open(csv_path, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

    print(f'Finished evaluating {len(pano_ids)} panos.')
    # Once all the panos have been evaluated, open csv_path and calculate the mean metrics across all panos
    # Exclude the first column, which is the pano_id.
    csv_path = os.path.join(directory, "mean_results.csv")
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['pano_id'])
    # Calculate the mean across all panos
    mean_df = df.mean(axis=0)
    # Make the first column the header and the second column the mean values
    mean_df = pd.DataFrame(mean_df).T
    # Make a new csv file called averaged_results.csv, but maintain the same header without the pano_id column
    mean_df.to_csv(os.path.join(directory, "averaged_results.csv"), index=False)

    # Save results as a pickle file
    with open(os.path.join(directory, "results.pickle"), 'wb') as f:
        pickle.dump(results, f)

    # Save the dictionary results in a json file, if it doesn't exist
    if not os.path.exists(os.path.join(directory, "results.json")):
        with open(os.path.join(directory, "results.json"), 'w') as f:
            # Convert the arrays in results to lists, so that they can be saved in a json file.
            # The following are the keys that need to be converted:
            # [pano_id][face_idx][closest_points][metrics][distances]
            # [pano_id][face_idx][closest_points][points]
            # [pano_id][face_idx][anchor_points][metrics][distances]
            # [pano_id][face_idx][anchor_points][points]
            for pano_id in results:
                for face_idx in results[pano_id]:
                    results[pano_id][face_idx]['closest_points']['metrics']['distances'] = results[pano_id][face_idx]['closest_points']['metrics']['distances'].tolist()
                    results[pano_id][face_idx]['closest_points']['points'] = results[pano_id][face_idx]['closest_points']['points'].tolist()
                    results[pano_id][face_idx]['anchor_points']['metrics']['distances'] = results[pano_id][face_idx]['anchor_points']['metrics']['distances'].tolist()
                    results[pano_id][face_idx]['anchor_points']['points'] = results[pano_id][face_idx]['anchor_points']['points'].tolist()
            json.dump(results, f)
    
    return results

def visualize(args, results, directory):
    # Iterate through the results dictionary
    '''The structure of results is:
    results = {
        'pano_id_1': {
                'face_idx_0': {
                    'closest_points': {
                        'metrics': {
                            'distances': closest_distances,
                            'precision': cp_precision,
                            'recall': cp_recall,
                            'f1': cp_f1,
                            'average_precision': cp_ap,
                        },
                        'points': closest_points,
                        'indices': {
                            'mask_indices': cp_mask_indices,
                            'gt_indices': cp_gt_indices
                        }
                    },
                    'anchor_points': {
                        'metrics': {
                            'distances': anchor_distances,
                            'precision': ap_precision,
                            'recall': ap_recall,
                            'f1': ap_f1,
                            'average_precision': ap_ap,
                        },
                        'points': anchor_points,
                        'indices': {
                            'mask_indices': ap_mask_indices,
                            'gt_indices': ap_gt_indices
                        }
                    },
                    'iou_metrics': {
                        'best_ious': best_ious,
                        'ap_50': ap50,
                        'ap_75': ap75
                    },
                    'iou_indices': {
                        'mask_indices': mask_indices_iou,
                        'gt_indices': gt_indices_iou
                    }
                }
                'face_idx_1' {...},
            },
        'pano_id_2': {...},
        ...
    }'''
    
    # Load labels from .csv file
    labels_df = pd.read_csv(args.labels_csv_path)

    # Load masks from .json file
    json_path = os.path.join(args.masks_dir, args.input_coco_format)
    with open(json_path) as f:
        panos_info = json.load(f)

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
    
    for pano_id in tqdm(results.keys()):
        '''1. Overlay the entire mask over the faces of the pano containing labels.'''
        '''2. Overlay the closest mask to the labels based on the closest points.'''
        '''3. Overlay the closest mask to the labels based on the anchor points.'''
        '''4. Overlay the best mask to the labels based on the IoU metric.'''

        # Create a directory for the pano
        visualize_pano_dir = os.path.join(directory, pano_id)
        if not os.path.exists(visualize_pano_dir):
            os.makedirs(visualize_pano_dir)

        # 1. Overlay the entire mask over the faces of the pano containing labels.
        visualize_labels(args, results[pano_id], pano_id, grouped_panos[pano_id], labels_df, visualize_pano_dir)

        # 2. Overlay the closest mask to the labels based on the closest points.
        visualize_closest_mask(args, results[pano_id], pano_id, grouped_panos[pano_id], labels_df, visualize_pano_dir, False)

        # 3. Overlay the closest mask to the labels based on the anchor points.
        visualize_closest_mask(args, results[pano_id], pano_id, grouped_panos[pano_id], labels_df, visualize_pano_dir, True)

        # 4. Overlay the best mask to the labels based on the IoU metric.
        visualize_best_iou_mask(args, results[pano_id], pano_id, grouped_panos[pano_id], labels_df, visualize_pano_dir)

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

    # Check if the .csv file containing the metrics already exists. If it doesn't,
    # evaluate the panos.
    if not os.path.exists(os.path.join(directory, "mean_results.csv")):
        results = evaluate(args, directory)
    else:
        print('Results already exist. Loading them...')
        # If it exists, visualize the results by loading "results.json"
        with open(os.path.join(directory, "results.json"), 'r') as f:
            results = json.load(f)
        # Create a directory to save the visualizations
        visualize_dir = os.path.join(directory, "visualize")
        visualize(args, results, visualize_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='res/dataset')
    parser.add_argument('--masks_dir', type=str, default='masks')
    parser.add_argument('--input_coco_format', type=str, default='input_coco_format.json')
    parser.add_argument('--labels_csv_path', type=str, default='res/labels/labels.csv')
    parser.add_argument('--threshold', type=int, default=100, help='Threshold for distance-based metrics (in pixels)')
    parser.add_argument('--radius', type=int, default=100, help='Radius for point-to-mask best IoU (in pixels)')
    parser.add_argument('--output_dir', type=str, default='evaluation')

    args = parser.parse_args()

    main(args)