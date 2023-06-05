''' This script contains functions for evaluating the performance of the pipeline
on a dataset of images. It includes functions for computing metrics such as precision, 
recall, and F1 score, as well as functions for visualizing the results of the evaluation. 
The script also includes functions for loading and processing the dataset, as well as 
for mapping face indices to the corresponding faces.
'''
import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from tqdm import tqdm
from scipy.spatial.distance import cdist
import argparse
import csv
import random
import cv2
import matplotlib.pyplot as plt
import pickle
import torch
from torchvision.ops import box_iou
from sklearn.metrics import auc, precision_recall_curve, average_precision_score

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

def get_bbox(mask):
    """Function to convert a mask to a bounding box."""
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    return [xmin, ymin, xmax, ymax]

def best_and_all_iou(pred_masks, bboxes):
    best_ious = []
    best_mask_indices = []
    bboxes_indices = []

    # create additional lists for all mask IoUs and mask indices 
    # (all_bboxes_indices will be the same as bboxes_indices)
    all_ious = []
    all_mask_indices = []

    for bbox_idx, bbox in enumerate(bboxes):
        gt_bbox = torch.tensor([bbox], dtype=torch.float)

        max_iou = -1
        best_mask_idx = -1

        for mask_idx, mask_dict in enumerate(pred_masks):
            rle = mask_dict['segmentation']
            pred_mask = mask_util.decode(rle)
            pred_bbox = get_bbox(pred_mask)
            if pred_bbox is None:
                print(f'Couldn\'t get bbox for mask {mask_idx}')
                continue
            pred_bbox = torch.tensor([pred_bbox], dtype=torch.float)

            # Calculate IoU using PyTorch function
            iou = box_iou(gt_bbox, pred_bbox).item()

            # append to all masks IoUs, indices and bbox indices
            all_ious.append(iou)
            all_mask_indices.append(mask_idx)

            if iou > max_iou:
                max_iou = iou
                best_mask_idx = mask_idx

        best_ious.append(max_iou)
        best_mask_indices.append(best_mask_idx)
        bboxes_indices.append(bbox_idx)

    return best_ious, best_mask_indices, bboxes_indices, all_ious, all_mask_indices

def precision_recall_f1(best_iou_scores, best_gt_indices, all_iou_scores, all_mask_indices, threshold):
    # Initialize counters
    true_positives = 0
    false_positives = 0  # don't consider all as false positives initially

    # Ensure best_iou_scores and best_gt_indices are the same length
    assert len(best_iou_scores) == len(best_gt_indices)

    # Keep track of ground truth indices that have been matched
    matched_gt_indices = set()

    # Iterate through each best IoU score and corresponding ground truth index
    for iou_score, gt_idx in zip(best_iou_scores, best_gt_indices):
        if iou_score >= threshold:
            true_positives += 1
            matched_gt_indices.add(gt_idx)

    # Now consider the false positives for all masks
    for iou_score, mask_idx in zip(all_iou_scores, all_mask_indices):
        if iou_score < threshold:
            false_positives += 1

    # Count false negatives
    all_gt_indices = set(best_gt_indices)
    false_negatives = len(all_gt_indices) - len(matched_gt_indices)

    # Compute precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # Return precision, recall, and F1 score
    return precision, recall, f1_score

def average_precision(all_iou_scores, threshold):
    # Initialize true and false labels
    true_ious = [1 if iou_score >= threshold else 0 for iou_score in all_iou_scores]
    false_ious = [1 if iou_score < threshold else 0 for iou_score in all_iou_scores]

    # If there are no true labels, return 0 for AP, precision, and recall
    if not any(true_ious):
        return 0.0

    # Sort by scores
    sorted_indices = np.argsort(all_iou_scores)[::-1]
    sorted_true_ious = np.array(true_ious)[sorted_indices]
    sorted_false_ious = np.array(false_ious)[sorted_indices]

    tp = sorted_true_ious.tolist()
    fp = sorted_false_ious.tolist()

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / len(true_ious)

    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    i = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])

    print(f'==== PRECISION, RECALL, AP@{threshold} ====')
    print(f'True labels: {true_ious}')
    print(f'IoU scores: {all_iou_scores}')
    print(f'Precision: {precisions}')
    print(f'Recall: {recalls}')
    print(f'AP: {ap}')
    print(f'=============================')
    return ap


def evaluate_single_face(masks, bboxes, empty_panos):
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
    print(f"Number of bboxes: {len(bboxes)}")
    n_masks = len(masks)
    print(f'Number of masks: {n_masks}')
    
    # Compute metrics
    # If there are no bounding boxes, all masks are false positives
    if (pano_id, face_idx) in empty_panos:
        result = {
            'empty': True,  # Indicate this image was empty
            'metrics': {
                'best_ious': [],
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'ap_35': 0,
                'ap_50': 0,
                'ap_75': 0,
            },
            'indices': {
                'mask_indices': [],
                'gt_indices': []
            }
        }
    else:
        # IoU
        best_ious, best_mask_indices_iou, bboxes_indices_iou, all_ious, all_mask_indices = best_and_all_iou(masks, bboxes)
        # Metrics 3: Precision, Recall, F1 @threshold
        precision, recall, f1 = precision_recall_f1(best_ious, bboxes_indices_iou, all_ious, all_mask_indices, args.threshold)
        # Metrics 4: Average Precision @35, @50 and @75
        ap05 = average_precision(all_ious, threshold=0.05)
        ap35 = average_precision(all_ious, threshold=0.35)
        ap50 = average_precision(all_ious, threshold=0.5)
        ap75 = average_precision(all_ious, threshold=0.75)

        result = {
            'empty': False,  # Indicate this image was not empty
            'metrics': {
                'best_ious': best_ious,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'ap_05': ap05,
                'ap_35': ap35,
                'ap_50': ap50,
                'ap_75': ap75,
            },
            'indices': {
                'mask_indices': best_mask_indices_iou,
                'gt_indices': bboxes_indices_iou
            }
        }

    print(result)
        
    return result

def evaluate(args, directory):

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

    # Load bboxes from .csv file
    bboxes = pd.read_csv(args.bboxes_path)

    # Count number of unique panos (pano is a dictionary)
    pano_ids = set(pano['pano_id'] for pano in panos_info)
    num_panoramas = len(pano_ids)
    print(f'Number of panos: {num_panoramas}')
    # Count number of masks (panos_info['pano_id']['face_idx']['masks'])
    print(f'Number of masks: {len(panos_info)}')
    # Count number of panos with bounding boxes
    bboxes_annotations = os.listdir(args.bboxes_dir)
    print(f'Number of bounding boxes: {len(bboxes_annotations)}')

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
    empty_panos = [] # list of panos with no ground truth bounding boxes

    pano_counter = 0
    for pano_id in tqdm(grouped_panos):
        if pano_counter == 1:
            break

        if pano_id not in results:
            results[pano_id] = {}

        if pano_id in bboxes_annotations:
            print(f'Processing pano {pano_id}:')
            for face_idx in grouped_panos[pano_id]:
                print(f'Number of masks for face {face_idx}: {len(grouped_panos[pano_id][face_idx])}')
                # Print number of bboxes for this pano and face
                # The structure of bboxes is a PD dataframe with the following columns: pano_id, face_idx, bboxes
                # The column bboxes contains a list of bboxes for each pano: [[xmin,ymin,xmax,ymax], [], ...]
                bboxes_face = bboxes[(bboxes['pano_id'] == pano_id) & (bboxes['face_idx'] == face_idx)]['bboxes'].values[0]
                # The problem is that bboxes_face is a string, not a list
                # Convert it to a list
                bboxes_face = eval(bboxes_face)
                # Convert the string inside the list to float
                bboxes_face = [[int(float(x)) for x in bbox] for bbox in bboxes_face]
                print(f'Number of bboxes for face {face_idx}: {len(bboxes_face)}')

                if len(bboxes_face) == 0:
                    print('The face does not contain obstacles.')
                    empty_panos.append((pano_id, face_idx))
                
                masks = grouped_panos[pano_id][face_idx]

                if face_idx not in results[pano_id]:
                    results[pano_id][face_idx] = {}

                eval_dict = evaluate_single_face(masks, bboxes_face, empty_panos)

                if eval_dict['empty']:
                    continue
                else:
                    results[pano_id][face_idx] = eval_dict
                    
                pano_counter += 1
                

            # Initialize accumulators for each metric across all faces
            total_best_iou = []
            total_precision = []
            total_recall = []
            total_f1 = []
            total_ap05 = []
            total_ap35 = []
            total_ap50 = []
            total_ap75 = []

            # Iterate through the results dictionary
            for pano_id, faces in results.items():
                if pano_id not in mean_results:
                    mean_results[pano_id] = {}
                
                for face_idx, metrics in faces.items():
                    # check if 'metrics' key exists in the dictionary
                    if 'metrics' in metrics:  
                        # Accumulate metrics for each face
                        total_best_iou.extend(metrics['metrics']['best_ious'])
                        total_precision.append(metrics['metrics']['precision'])
                        total_recall.append(metrics['metrics']['recall'])
                        total_f1.append(metrics['metrics']['f1'])
                        total_ap35.append(metrics['metrics']['ap_35'])
                        total_ap50.append(metrics['metrics']['ap_50'])
                        total_ap75.append(metrics['metrics']['ap_75'])

            # Calculate the mean for each metric across all faces
            mean_best_iou = np.mean(total_best_iou)
            mean_precision = np.mean(total_precision)
            mean_recall = np.mean(total_recall)
            mean_f1 = np.mean(total_f1)
            mean_ap05 = np.mean(total_ap05)
            mean_ap35 = np.mean(total_ap35)
            mean_ap50 = np.mean(total_ap50)
            mean_ap75 = np.mean(total_ap75)

            # Store the mean metrics in the mean_results dictionary
            mean_results[pano_id] = {
                'mean_best_iou': mean_best_iou,
                'mean_precision': mean_precision,
                'mean_recall': mean_recall,
                'mean_f1': mean_f1,
                'mean_ap05': mean_ap05,
                'mean_ap35': mean_ap35,
                'mean_ap50': mean_ap50,
                'mean_ap75': mean_ap75
            }

            # Set the filename for the CSV file
            csv_path = os.path.join(directory, "bboxes_mean_results.csv")

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

    print(f'Finished evaluating {pano_counter} panos.')
    # Once all the panos have been evaluated, open csv_path and calculate the mean metrics across all panos
    # Exclude the first column, which is the pano_id.
    csv_path = os.path.join(directory, "bboxes_mean_results.csv")
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['pano_id'])
    # Calculate the mean across all panos
    mean_df = df.mean(axis=0)
    # Make the first column the header and the second column the mean values
    mean_df = pd.DataFrame(mean_df).T
    # Make a new csv file called averaged_results.csv, but maintain the same header without the pano_id column
    mean_df.to_csv(os.path.join(directory, "bboxes_averaged_results.csv"), index=False)

    # Save results as a pickle file
    with open(os.path.join(directory, "bboxes_results.pickle"), 'wb') as f:
        pickle.dump(results, f)

    # Save the dictionary results in a json file, if it doesn't exist
    if not os.path.exists(os.path.join(directory, "bboxes_results.json")):
        with open(os.path.join(directory, "bboxes_results.json"), 'w') as f:
            # Convert the arrays in results to lists, so that they can be saved in a json file.
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
    print(f'Bboxes path: {args.bboxes_path}')
    print(f'Output directory: {directory}')

    # Check if the .csv file containing the metrics already exists. If it doesn't,
    # evaluate the panos.
    if not os.path.exists(os.path.join(directory, "bboxes_mean_results.csv")):
        results = evaluate(args, directory)
    else:
        print('Results already exist. Loading them...')
        # If it exists, visualize the results by loading "results.json"
        with open(os.path.join(directory, "bboxes_results.json"), 'r') as f:
            results = json.load(f)
        # Create a directory to save the visualizations
        visualize_dir = os.path.join(directory, "visualize")
        visualize(args, results, visualize_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='res/dataset')
    parser.add_argument('--masks_dir', type=str, default='masks')
    parser.add_argument('--input_coco_format', type=str, default='input_coco_format.json')
    parser.add_argument('--bboxes_path', type=str, default='res/labels/labels.csv')
    parser.add_argument('--bboxes_dir', type=str, default='res/labels')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str, default='evaluation')

    args = parser.parse_args()

    main(args)