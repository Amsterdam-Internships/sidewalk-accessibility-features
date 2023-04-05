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

def get_ps_labels():

    base_url = "https://sidewalk-amsterdam.cs.washington.edu/v2/access/attributesWithLabels?lat1={}&lng1={}&lat2={}&lng2={}" 
    
    whole = (52.303, 4.8, 52.425, 5.05)
    centrum_west = (52.364925, 4.87444, 52.388692, 4.90641)
    test = (52.0, 4.0, 53.0, 5.0)

    coords = test

    url = base_url.format(*coords)

    local_dump = url.replace('/', '|')

    try:
        project_sidewalk_labels = json.load(open(local_dump, 'r'))
    except Exception as e:
        print("Couldn't load local dump")
        project_sidewalk_labels = requests.get(url.format(*coords)).json()
        json.dump(project_sidewalk_labels, open(local_dump, 'w'))

    ps_labels_df = gpd.GeoDataFrame.from_features(project_sidewalk_labels['features'])
    # Print length before filtering
    print('Length raw labels: ', len(ps_labels_df))

    return ps_labels_df

def get_xy_coords_ps_labels():
    # Send a get call to this API: https://sidewalk-amsterdam-test.cs.washington.edu/adminapi/labels/cvMetadata
    other_labels = requests.get('https://sidewalk-amsterdam.cs.washington.edu/adminapi/labels/cvMetadata').json()
    other_labels_df = pd.DataFrame(other_labels)
    print('Length raw labels (2): ', len(other_labels_df))

    return other_labels_df

def visualize_labels(args, gt_points, pano_id, path):
    # Load the image
    image_path = os.path.join(args.input_dir, pano_id)
    backprojected_path = os.path.join(os.path.dirname(args.input_dir), 'backprojected')
    mask_path = os.path.join(backprojected_path, f'{pano_id}/{pano_id}.png')
    image = plt.imread(image_path)
    mask = plt.imread(mask_path)

    fig = plt.figure()
    plt.imshow(image, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.4)

    for gt_point in gt_points:
        plt.scatter(gt_point[1], gt_point[0], color='red', s=10)
    
    # Save the image
    fig.savefig(os.path.join(path, f'{pano_id}.png'), dpi=300, bbox_inches='tight')

def visualize_debug_mask(gt_points, pred_masks, distances, closest_points, gt_indices, pano_id, path):
    num_masks = len(pred_masks)
    fig, axes = plt.subplots(num_masks, 1, figsize=(5, 3 * num_masks))

    for idx, (pred_mask, ax) in enumerate(zip(pred_masks, axes)):
        ax.imshow(pred_mask)  # Display the predicted mask

        # Retrieve the correct ground truth point, distance, and closest point
        gt_point = gt_points[gt_indices[idx]]
        distance = distances[idx]
        closest_point = closest_points[idx]
        
        y, x = gt_point
        closest_y, closest_x = closest_point

        ax.scatter(x, y, c='red', marker='x', s=50, label='Ground Truth')  # Mark the ground truth point
        ax.plot([x, closest_x], [y, closest_y], 'r--', label='Distance')  # Draw the distance line
        ax.scatter(closest_x, closest_y, c='blue', marker='o', s=20, label='Closest Point')  # Mark the closest point

        ax.set_title(f"Closest mask-to-point distance for mask {idx + 1}")
        ax.legend()

    plt.tight_layout()
    plt.show()

    fig.savefig(os.path.join(path, f'{pano_id}_mask.png'), dpi=300, bbox_inches='tight')

def visualize_best_dilated(args, gt_points, pred_masks, best_gt_point_indices, pano_id, path):
    num_masks = len(pred_masks)
    fig, axes = plt.subplots(num_masks, 1, figsize=(5, 3 * num_masks))

    # Generate a grid of coordinates for the entire image
    y_coords, x_coords = np.indices(pred_masks[0].shape)
    coords = np.column_stack((y_coords.ravel(), x_coords.ravel()))

    for idx, (pred_mask, ax) in enumerate(zip(pred_masks, axes)):
        best_gt_point = gt_points[best_gt_point_indices[idx]]

        # Compute the distance from the best ground truth point to all other points
        distances = cdist([best_gt_point], coords).reshape(pred_mask.shape)

        # Create a dilated mask by thresholding the distance matrix at the specified radius
        gt_mask_dilated = (distances <= args.radius).astype(int)

        # Display the dilated ground truth mask
        ax.imshow(gt_mask_dilated, alpha=0.5, cmap='gray', label='Dilated Ground Truth')

        # Display the predicted mask
        ax.imshow(pred_mask, alpha=0.5, cmap='jet', label='Predicted Mask')

        # Mark the best ground truth point
        ax.scatter(best_gt_point[1], best_gt_point[0], c='red', marker='x', s=50, label='Ground Truth')

        ax.set_title(f"Best dilated ground truth point and mask {idx + 1}")
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Save the image
    fig.savefig(os.path.join(path, f'{pano_id}_mask_iou.png'), dpi=300, bbox_inches='tight')

def compute_label_coordinates(dataframe, pano_id):

    # Find all the labels in the panorama
    labels = dataframe[dataframe['gsv_panorama_id'] == pano_id]

    labels_coords = []

    # For each label, find its coordinates in the panorama
    for index, row in labels.iterrows():
        #print('Labels coming from panorama:', row['gsv_panorama_id'])
        label_name = row['label_id']
        image_width = row['image_width']
        image_height = row['image_height']
        sv_image_x = row['sv_image_x']
        # Momentarily change the definition of sv_image_y. For now,
        # sv_image_y is the offset w.r.t. the middle of the image
        # which means that the real y coordinate is y = (image_width / 2) - sv_image_y
        sv_image_y = row['sv_image_y']
        real_y = (image_height / 2) - sv_image_y

        #print(f'Original image width: {image_width}, height: {image_height}')
        #print(f'Original label {label_name} coordinates: {sv_image_x}, {sv_image_y}')
        #print(f'Original label {label_name} with real y-coordinates' \
        #    f'({image_height}/2){sv_image_y}: {sv_image_x}, {real_y}')

        # image_width : sv_image_x = my_pano_width : sv_pano_x
        # sv_pano_x = (sv_image_x * my_pano_width) / image_width
        sv_pano_x = int(sv_image_x*image.shape[1]/image_width)
        sv_pano_y = int(real_y*image.shape[0]/image_height)

        #print(f'Scaled label {label_name} coordinates: {sv_pano_x}, {sv_pano_y} \n')

        # Visualize the point (sv_pano_x, sv_pano_y) on the image
        #print(f'Label {label_name} coordinates: {sv_pano_x}, {sv_pano_y} \n')
        #plt.scatter(sv_pano_x, sv_pano_y, color='red', s=10)

        labels_coords.append((sv_pano_y, sv_pano_x))

    return labels_coords

def point_to_mask_distance(gt_points, pred_masks):
    distances = []
    closest_points = []
    mask_indices = []

    for gt_point in gt_points:
        min_distance = float('inf')
        closest_y, closest_x = -1, -1
        mask_index = -1

        for idx, pred_mask in enumerate(pred_masks):
            if pred_mask.ndim == 2:  # Check if the mask is 2D
                mask_channel = pred_mask
            elif pred_mask.ndim == 3:  # Check if the mask is 3D
                mask_channel = pred_mask[:, :, 0]  # Assuming the first channel contains the binary mask
            else:
                raise ValueError("Invalid mask dimensions")

            mask_coords = np.where(mask_channel > 0)  # Get the indices of non-zero elements (i.e., the mask)
            mask_coords = np.vstack(mask_coords).T  # Stack the mask indices into a 2D array

            point_coords = np.array(gt_point).reshape(1, -1)
            dist = cdist(point_coords, mask_coords)

            local_min_distance_idx = np.argmin(dist)
            local_min_distance = dist[0, local_min_distance_idx]
            local_closest_y, local_closest_x = mask_coords[local_min_distance_idx]

            if local_min_distance < min_distance:
                min_distance = local_min_distance
                closest_y, closest_x = local_closest_y, local_closest_x
                mask_index = idx

        distances.append(min_distance)
        closest_points.append((closest_y, closest_x))
        mask_indices.append(mask_index)

    return distances, closest_points, mask_indices

def mask_to_point_distance(gt_points, pred_masks):
    distances = []
    closest_points = []
    gt_indices = []

    for idx, pred_mask in enumerate(pred_masks):
        min_distance = float('inf')
        closest_y, closest_x = -1, -1
        gt_point_index = -1

        if pred_mask.ndim == 2:  # Check if the mask is 2D
            mask_channel = pred_mask
        elif pred_mask.ndim == 3:  # Check if the mask is 3D
            mask_channel = pred_mask[:, :, 0]  # Assuming the first channel contains the binary mask
        else:
            raise ValueError("Invalid mask dimensions")

        mask_coords = np.where(mask_channel > 0)  # Get the indices of non-zero elements (i.e., the mask)
        mask_coords = np.vstack(mask_coords).T  # Stack the mask indices into a 2D array

        for gt_idx, gt_point in enumerate(gt_points):
            point_coords = np.array(gt_point).reshape(1, -1)
            dist = cdist(point_coords, mask_coords)

            local_min_distance_idx = np.argmin(dist)
            local_min_distance = dist[0, local_min_distance_idx]
            local_closest_y, local_closest_x = mask_coords[local_min_distance_idx]

            if local_min_distance < min_distance:
                min_distance = local_min_distance
                closest_y, closest_x = local_closest_y, local_closest_x
                gt_point_index = gt_idx

        distances.append(min_distance)
        closest_points.append((closest_y, closest_x))
        gt_indices.append(gt_point_index)

    return distances, closest_points, gt_indices

def mask_to_point_best_iou(gt_points, pred_masks, radius=5):
    best_ious = []
    best_gt_point_indices = []

    # Generate a grid of coordinates for the entire image
    y_coords, x_coords = np.indices(pred_masks[0].shape)
    coords = np.column_stack((y_coords.ravel(), x_coords.ravel()))

    for idx, pred_mask in enumerate(pred_masks):
        max_iou = 0
        best_gt_point_index = -1

        for point_idx, gt_point in enumerate(gt_points):
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

    #print(f'Best IoUs: {best_ious}')
    #print(f'Best ground truth point indices: {best_gt_point_indices}')

    return best_ious, best_gt_point_indices

def precision_recall_f1(distances, gt_indices, threshold):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

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

    print(f"True positives: {true_positives}, False positives: {false_positives}, False negatives: {false_negatives}")

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

def average_precision_iou(best_ious, gt_points, threshold):
    num_gt_points = len(gt_points)
    num_pred_masks = len(best_ious)

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

def evaluate(args, directory):

    # Get Project Sidewalk labels from API
    ps_labels_df = get_ps_labels()
    # Get the labels from another API call
    other_labels_df = get_xy_coords_ps_labels()

    # Filter labels dataframe to only contain obstacles
    ps_labels_df = ps_labels_df[ps_labels_df['label_type'] == 'Obstacle']

    # Intersect other_labels_df with ps_labels_df
    other_labels_df = other_labels_df[other_labels_df['gsv_panorama_id'].isin(ps_labels_df['gsv_panorama_id'])]
    print('Number of labels in other_labels_df after filtering for obstacles: ', len(other_labels_df))

    # Load masks from .json file
    json_path = os.path.join(os.path.dirname(args.input_dir), 'backprojected')
    json_path = os.path.join(json_path, 'input_coco_format.json')
    with open(json_path) as f:
        data = json.load(f)

    # The structure of data is a list of dictionaries, with instance['pano_id'] as the panorama ID
    # and instance['segmentation'] as the corresponding masks. Make a dictionary of masks with
    # the panorama ID as the key and a list of masks as value.
    panos = {}
    for instance in data[50:80]:
        pano_id = instance['pano_id'].replace(".jpg", "")
        mask = mask_util.decode(instance['segmentation'])
        if pano_id not in panos:
            panos[pano_id] = []
        panos[pano_id].append(mask)

    avg_distance = []
    avg_iou = []
    avg_precision = []
    avg_recall = []
    avg_f1 = []
    avg_ap = []
    avg_ap50 = []
    avg_ap75 = []

    for pano in tqdm(panos):
        '''Assumptions: there is one ground truth for each mask, 
        and the order of the masks is the same as the order of the ground truth indices.'''
        if pano in other_labels_df["gsv_panorama_id"].values:
            print(f'Pano: {pano}')
            pred_masks = panos[pano]
            # Compute label coordinates for pano
            gt_points = compute_label_coordinates(other_labels_df, pano)
            n_masks = len(panos[pano])
            # Compute metrics
            # Metrics 1: (average) mask-to-point distance
            distances, closest_points, gt_indices = mask_to_point_distance(gt_points, pred_masks)
            mean_distance = np.mean(distances) # mean between all masks and points
            # Metrics 2: point-to-mask best IoU
            best_ious, best_gt_point_indices = mask_to_point_best_iou(gt_points, pred_masks, radius=100)
            mean_ious = np.mean(best_ious)
            # Metrics 3: Precision, Recall, F1 (already uses closest_points based on point-to-mask distance)
            precision, recall, f1 = precision_recall_f1(distances, gt_indices, args.threshold)
            # Metrics 4: Average precision based on point-to-mask distance
            ap = average_precision(distances, gt_indices, args.threshold)
            # Metrics 5: Average precision @50 and @75 based on IoU
            # Print best ious with 3 decimals, considering that it's a list of numbers
            ap_50 = average_precision_iou(best_ious, gt_points, threshold=0.5)
            ap_75 = average_precision_iou(best_ious, gt_points, threshold=0.75)
            # We can't do mAP because at the moment there is only one class (obstacles)

            '''# Print all the metrics in a pretty way
            print(f"\n")
            print(f"==== Printing metrics for pano {pano}: ====")
            print(f"Average point-to-mask distance: {avg_distance:.3f}")
            print(f"Average best IoU: {avg_ious:.3f}")
            print(f"== Distance-based metrics, threshold: 100 ==")
            print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            print(f"Average Precision: {ap:.3f}")
            print(f"== IoU-based metrics, threshold: 0.5 ==")
            print(f"Average Precision @50 (IoU): {ap_50:.3f}")
            print(f"Average Precision @75 (IoU): {ap_75:.3f}")'''

            if args.visualize:
                # Probability of visualization: 1% of the panos
                if random.random() > 0.01:
                    continue
                else:
                    visualization_path = os.path.join(os.path.dirname(args.input_dir), 'visualized')
                    print(f'Visualization of pano {pano} with threshold=', args.threshold, ' and radius=', args.radius, '')
                    # Make a folder 'visualized' if not present
                    if not os.path.exists(visualization_path):
                        os.makedirs(visualization_path)
                    visualize_labels(args, gt_points, pano_id, visualization_path) # visualize labels and masks on pano
                    visualize_debug_mask(gt_points, pred_masks, distances, \
                                        closest_points, gt_indices, pano, visualization_path) # metrics 1
                    visualize_best_dilated(args, gt_points, pred_masks, best_gt_point_indices, \
                                            pano, visualization_path) # metrics 2

            # Append metrics to lists
            avg_distance.append(mean_distance)
            avg_iou.append(mean_ious)
            avg_precision.append(precision)
            avg_recall.append(recall)
            avg_f1.append(f1)
            avg_ap.append(ap)
            avg_ap50.append(ap_50)
            avg_ap75.append(ap_75)

            # Save metrics to .csv file, adding as first column the panorama ID
            metrics = [pano, mean_distance, mean_ious, precision, recall, f1, ap, ap_50, ap_75]
            with open(os.path.join(directory, 'metrics.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(metrics)
    # Last row of the .csv file is the average of all the metrics
    avg_metrics = ['Average', np.mean(avg_distance), np.mean(avg_iou), np.mean(avg_precision), np.mean(avg_recall), np.mean(avg_f1), np.mean(avg_ap), np.mean(avg_ap50), np.mean(avg_ap75)]
    with open(os.path.join(directory, f'metrics_{args.threshold}_{args.radius}.csv'), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(avg_metrics)

def main(args):
    # Replace everything that is not a character with an underscore in neighbourhood string, and make it lowercase
    args.neighbourhood = re.sub(r'[^a-zA-Z]', '_', args.neighbourhood).lower()

    args.input_dir = os.path.join(args.input_dir, args.neighbourhood)
    args.input_dir = args.input_dir + '_' + args.quality
    args.input_dir = os.path.join(args.input_dir, 'reoriented')

    # Create the output directory if it doesn't exist
    directory = os.path.join(os.path.dirname(args.input_dir), 'evaluation')

    if not os.path.exists(directory):
        os.makedirs(directory)

    evaluate(args, directory)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='res/dataset')
    parser.add_argument('--neighbourhood', type=str, default='osdorp')
    parser.add_argument('--quality', type=str, default='full')
    parser.add_argument('--visualize', type=bool, default=False)
    parser.add_argument('--threshold', type=int, default=100, help='Threshold for distance-based metrics (in pixels)')
    parser.add_argument('--radius', type=int, default=100, help='Radius for point-to-mask best IoU (in pixels)')

    args = parser.parse_args()

    main(args)