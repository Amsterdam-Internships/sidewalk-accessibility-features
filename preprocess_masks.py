import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import json
import pycocotools.mask as mask_util
import concurrent.futures
import psutil
import sys
import csv

sys.settrace

def reverse_map_faces(face):
    # Function that maps the face name to the corresponding index
    # Mapping: 'front' -> 0, 'right' -> 1, 'back' -> 2, 'left' -> 3, 'top' -> 4, 'bottom' -> 5
    face_mapping = {'front': 0, 'right': 1, 'back': 2, 'left': 3, 'top': 4, 'bottom': 5}
    return face_mapping.get(face)

def map_faces(face_idx):
    # Function that maps the face index to the corresponding name
    # Mapping: 0 -> 'front', 1 -> 'right', 2 -> 'back', 3 -> 'left', 4 -> 'top', 5 -> 'bottom'
    face_mapping = {0: 'front', 1: 'right', 2: 'back', 3: 'left', 4: 'top', 5: 'bottom'}
    return face_mapping.get(face_idx)

def resize_masks(args):
    print(f'Resizing masks to {args.size} x {args.size} pixels...')
    # args.input_dir contains the pano folders with the masks
    # args.output_dir will contain the resized masks
    directory = args.output_dir

    # For each pano folder in args.input_dir, go through the masks, resize them and save them in output_dir/pano_folder
    for pano in tqdm(os.listdir(args.input_dir)):
        pano_masks_path = os.path.join(args.input_dir, pano)
        pano_masks = os.listdir(pano_masks_path)

        # Create the pano folder in the output directory if it doesn't exist
        if not os.path.exists(os.path.join(directory, pano)):
            os.makedirs(os.path.join(directory, pano))

        for pano_mask in pano_masks:
            # Check if there is already a resized mask in the output directory
            if os.path.exists(os.path.join(directory, pano, pano_mask)):
                continue

            # Load the mask
            mask = cv2.imread(os.path.join(pano_masks_path, pano_mask), cv2.IMREAD_GRAYSCALE)

            # Check if the mask is already the desired size
            if mask.shape[0] == args.size and mask.shape[1] == args.size:
                continue

            # Resize the mask
            mask = cv2.resize(mask, (args.size, args.size))

            # Save the mask
            cv2.imwrite(os.path.join(directory, pano, pano_mask), mask)
    print('Done!')

def filter_masks(args):
    '''The panos_info dictionary will have the following structure:
        panos_info = {
            'pano_id': {
                'face_idx': {
                    'masks': [...],
                    'areas': [...],
                    'centroids': [...]
                },
                ...
            },
            ...
        }'''
    print('Applying connected component analysis and filtering...')
    directory = args.output_dir
    
    # Initialize the dictionary to store panos information
    panos_info = {}

    # Initialize list of panos (remove .json and .csv files)
    panos_list = [pano for pano in os.listdir(directory) if not pano.endswith('.json') and not pano.endswith('.csv')]

    def process_pano(pano):
        # For each pano folder in args.output_dir, go through the masks, apply connected component analysis and filter the masks
        pano_masks_path = os.path.join(directory, pano)
        pano_masks = os.listdir(pano_masks_path)

        # Initialize a dictionary to store face indices, masks, areas, and centroids for the current pano
        pano_data = {}

        for pano_mask in pano_masks:
            
            # Get the face index from the pano_mask filename
            face_name = os.path.splitext(pano_mask)[0]
            face_idx = reverse_map_faces(face_name)

            # Load the mask
            mask = cv2.imread(os.path.join(pano_masks_path, pano_mask), cv2.IMREAD_GRAYSCALE)

            # Threshold the image
            _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Apply connected component analysis
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

            # Remove the background
            output[output == 0] = -1
            output -= 1
            stats = stats[1:]
            centroids = centroids[1:]
            nb_components -= 1

            # Minimum size of particles we want to keep (number of pixels)
            min_size = args.threshold

            # Your answer image
            filtered_mask = np.zeros((output.shape))

            # Initialize lists to store masks, areas, and centroids for the current face index
            masks = []
            areas = []
            centroids_list = []

            # For every component in the image, keep it only if it's above min_size
            for i in range(0, nb_components):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_size:
                    component_mask = np.zeros((output.shape))  # Create a new mask for the current component
                    component_mask[output == i] = 255
                    masks.append(component_mask)  # Append the new mask to the list
                    areas.append(area)
                    centroids_list.append(centroids[i])

            # Store the information for the current face index in the pano_data dictionary
            pano_data[face_idx] = {
                'masks': masks,
                'areas': areas,
                'centroids': centroids_list
            }

        # Store the information for the current pano in the panos_info dictionary
        panos_info[pano] = pano_data

    # Calculate max_threads based on CPU capacity
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent()
    max_threads = int((cpu_count * (1 - cpu_percent / 100)) * 0.2)

    # Use ThreadPoolExecutor to limit the number of concurrent threads
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        list(tqdm(executor.map(process_pano, panos_list), total=len(panos_list))) 

    print('Done!')
    return panos_info

def filter_masks_by_bboxes(args, panos_info):
    '''Load the sidewalk bounding boxes from the corresponding .json file
    and filter the masks in panos_info that are not inside the bounding boxes.'''
    seg_dir = args.seg_dir

    print(f'Filtering masks by sidewalk bounding boxes...')

    # Create a new panos_info dictionary with the same structure as the original one
    new_panos_info = {}
    for pano_id in panos_info:
        new_panos_info[pano_id] = {}
        for face_idx in panos_info[pano_id]:
            new_panos_info[pano_id][face_idx] = {
                'masks': [],
                'areas': [],
                'centroids': []
            }

    # Create a list to keep track of the pano_id and face_idx where there is no 'sidewalk' bbox in the .json file
    no_sidewalk_bbox_list = []

    for pano_id in tqdm(panos_info):
        pano_dir = os.path.join(seg_dir, pano_id)

        for face_idx in panos_info[pano_id]:
            # Load the bounding boxes from the corresponding .json file.
            # The file is called 'mask_{face_idx}.json'
            # If it doesn't exist, skip the current face index
            # The structure of the .json file is:
            '''[
                {"value": 0, "label": "background"},
                {"value": 1, "label": "sidewalk", "logit": 0.4, "box": [1.101593017578125, 525.6083984375, 1022.8046875, 1021.849609375]}, 
                {"value": 2, "label": "road", "logit": 0.36, "box": [758.01318359375, 515.32080078125, 1023.3589477539062, 654.1165771484375]}
            ]'''
            # The value and the associated label are dynamic. The value 0 is always associated with the background.
            # Filter the masks that are not inside the bounding boxes
            try:
                face_name = map_faces(face_idx)
                with open(os.path.join(pano_dir, f'mask_{face_name}.json'), 'r') as f:
                    data = json.load(f)
            except FileNotFoundError:
                # Copy the masks, areas, and centroids from panos_info to new_panos_info
                print(f'File not found for pano {pano_id} and face {face_idx}. Skipping...')
                new_panos_info[pano_id][face_idx]['masks'] = panos_info[pano_id][face_idx]['masks']
                new_panos_info[pano_id][face_idx]['areas'] = panos_info[pano_id][face_idx]['areas']
                new_panos_info[pano_id][face_idx]['centroids'] = panos_info[pano_id][face_idx]['centroids']
                continue

            sidewalk_found = False
            # Iterate over each bounding box in data that has the label 'sidewalk'
            # If there is no 'sidewalk' label, look for 'sidewalkroad' label
            # If there are no 'sidewalk' or 'sidewalkroad' labels, skip the current face index
            for d in data:
                #print(f'Label: {d.get("label")}')
                # d has the following structure: {"value", "label", "logit", "box": []}
                # Iterate over the dictionary keys
                if d.get('label') in ['sidewalk', 'sidewalkroad']:
                    #print(f'bbox: {d.get("box")}')
                    sidewalk_found = True
                    bbox = d.get('box')

                    # Iterate over each mask in panos_info[pano_id][face_idx]['masks']
                    for mask, area, centroid in zip(panos_info[pano_id][face_idx]['masks'], panos_info[pano_id][face_idx]['areas'], panos_info[pano_id][face_idx]['centroids']):
                        bbox_mask = np.zeros((mask.shape))
                        bbox_mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 255

                        if np.any(np.logical_and(mask, bbox_mask)):
                            # Append the new mask, area, and centroid to the corresponding lists in the new_panos_info dictionary
                            new_panos_info[pano_id][face_idx]['masks'].append(mask)
                            new_panos_info[pano_id][face_idx]['areas'].append(area)
                            new_panos_info[pano_id][face_idx]['centroids'].append(centroid)

            if not sidewalk_found:
                # If there are no 'sidewalk' or 'sidewalkroad' labels, copy
                # the masks, areas, and centroids from panos_info to new_panos_info
                #print(f'No sidewalk or sidewalkroad labels found for pano {pano_id} and face {face_name}. Skipping...')
                no_sidewalk_bbox_list.append((pano_id, face_name))
                new_panos_info[pano_id][face_idx]['masks'] = panos_info[pano_id][face_idx]['masks']
                new_panos_info[pano_id][face_idx]['areas'] = panos_info[pano_id][face_idx]['areas']
                new_panos_info[pano_id][face_idx]['centroids'] = panos_info[pano_id][face_idx]['centroids']

    # Iterate over each pano_id in panos_info
    total_masks1 = 0
    total_masks2 = 0
    for pano_id in panos_info:
        # Iterate over each face_idx in the current pano_id
        for face_idx in panos_info[pano_id]:
            # Get the number of dictionaries in the current face_idx of panos_info
            num_dicts1 = len(panos_info[pano_id][face_idx]['masks'])
            total_masks1 += num_dicts1
            # Get the number of dictionaries in the current face_idx of new_panos_info
            num_dicts2 = len(new_panos_info[pano_id][face_idx]['masks'])
            total_masks2 += num_dicts2
            
    # Print the total number of masks across the two dictionaries
    print(f"Total number of masks before filtering: {total_masks1}. \n Total \
          number of masks after filtering: {total_masks2}")
    print('Done!')

    return new_panos_info, no_sidewalk_bbox_list

def filter_masks_by_masks(args, panos_info):
    '''Load the sidewalk masks from the corresponding .json file and filter the dilated masks in panos_info
    that result in an empty set when performing a logical AND operation between masks.'''
    seg_dir = args.seg_dir

    print(f'Filtering object masks by sidewalk masks...')

    # Create a new panos_info dictionary with the same structure as the original one
    new_panos_info = {}
    for pano_id in panos_info:
        new_panos_info[pano_id] = {}
        for face_idx in panos_info[pano_id]:
            new_panos_info[pano_id][face_idx] = {
                'masks': [],
                'areas': [],
                'centroids': []
            }

    for pano_id in tqdm(panos_info):
        pano_dir = os.path.join(seg_dir, pano_id)

        for face_idx in panos_info[pano_id]:
            # Load the bounding boxes from the corresponding .json file.
            # The file is called 'mask_{face_idx}.json'
            # If it doesn't exist, skip the current face index
            # The structure of the .json file is:
            '''[
                {"value": 0, "label": "background"},
                {"value": 1, "label": "sidewalk", "logit": 0.4, "box": [1.101593017578125, 525.6083984375, 1022.8046875, 1021.849609375]}, 
                {"value": 2, "label": "road", "logit": 0.36, "box": [758.01318359375, 515.32080078125, 1023.3589477539062, 654.1165771484375]}
            ]'''
            # The value and the associated label are dynamic. The value 0 is always associated with the background.
            # Filter the masks that are not inside the bounding boxes
            try:
                with open(os.path.join(pano_dir, f'mask_{face_idx}.json'), 'r') as f:
                    data = json.load(f)
            except FileNotFoundError:
                # Copy the masks, areas, and centroids from panos_info to new_panos_info
                print(f'File not found for pano {pano_id} and face {face_idx}. Skipping...')
                new_panos_info[pano_id][face_idx]['masks'] = panos_info[pano_id][face_idx]['masks']
                new_panos_info[pano_id][face_idx]['areas'] = panos_info[pano_id][face_idx]['areas']
                new_panos_info[pano_id][face_idx]['centroids'] = panos_info[pano_id][face_idx]['centroids']
                continue

            # Iterate over each mask in panos_info[pano_id][face_idx]['masks']
            for mask, area, centroid in zip(panos_info[pano_id][face_idx]['masks'], panos_info[pano_id][face_idx]['areas'], panos_info[pano_id][face_idx]['centroids']):
                # Iterate over each bounding box in data that has the label 'sidewalk'
                # If there is no 'sidewalk' label, look for 'sidewalkroad' label
                # If there are no 'sidewalk' or 'sidewalkroad' labels, skip the current face index
                for d in data:
                    # d has the following structure: {"value", "label", "logit", "box": []}
                    # Iterate over the dictionary keys
                    if d.get('label') in ['sidewalk', 'sidewalkroad']:
                        bbox = d.get('box')
                        bbox_mask = np.zeros((mask.shape))
                        bbox_mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 255
                        if np.any(np.logical_and(mask, bbox_mask)):
                            # Append the new mask, area, and centroid to the corresponding lists in the new_panos_info dictionary
                            new_panos_info[pano_id][face_idx]['masks'].append(mask)
                            new_panos_info[pano_id][face_idx]['areas'].append(area)
                            new_panos_info[pano_id][face_idx]['centroids'].append(centroid)
                else:
                    # If there are no 'sidewalk' or 'sidewalkroad' labels, copy
                    # the masks, areas, and centroids from panos_info to new_panos_info
                    print('No sidewalk or sidewalkroad labels found. Skipping...')
                    new_panos_info[pano_id][face_idx]['masks'] = panos_info[pano_id][face_idx]['masks']
                    new_panos_info[pano_id][face_idx]['areas'] = panos_info[pano_id][face_idx]['areas']
                    new_panos_info[pano_id][face_idx]['centroids'] = panos_info[pano_id][face_idx]['centroids']

    # Iterate over each pano_id in panos_info
    total_masks1 = 0
    total_masks2 = 0
    for pano_id in panos_info:
        # Iterate over each face_idx in the current pano_id
        for face_idx in panos_info[pano_id]:
            # Get the number of dictionaries in the current face_idx of panos_info
            num_dicts1 = len(panos_info[pano_id][face_idx]['masks'])
            total_masks1 += num_dicts1
            # Get the number of dictionaries in the current face_idx of new_panos_info
            num_dicts2 = len(new_panos_info[pano_id][face_idx]['masks'])
            total_masks2 += num_dicts2
            
    # Print the total number of masks across the two dictionaries
    print(f"Total number of masks before filtering: {total_masks1}. \
           Total number of masks after filtering: {total_masks2}")
    print('Done!')
    return new_panos_info

def save_masks(args, panos_info):
    directory = args.output_dir

    # If input_coco_format.json exists, load it
    if os.path.exists(os.path.join(directory, f'input_coco_format_{args.threshold}.json')):
        with open(os.path.join(directory, f'input_coco_format_{args.threshold}.json'), 'r') as f:
            input_coco_format = json.load(f)
    else:
        input_coco_format = []

    # The panos we want to process are the ones we have their masks
    print(f'Looking for masks in panos_info')
    panos = list(panos_info.keys())

    # Print number of panos before filtering
    print('Number of panos before filtering:', len(panos))

    # Skip the ones we already processed
    panos = [pano for pano in panos if pano not in [pano['pano_id'] for pano in input_coco_format]]
    # Print number of panos after filtering
    print('Number of panos after filtering:', len(panos))

    def process_pano(pano):
        #print('Processing panorama', pano)

        # Check if there are less than 4 unique face indices.
        # If so, skip this panorama and save pano in a new .csv
        unique_faces = len(set(panos_info[pano].keys()))
        if unique_faces < 4:
            print('Less than 4 unique face indices found, skipping panorama')
            with open(os.path.join(directory, f'incomplete_panos.csv'), 'a') as f:
                f.write(f'{pano}\n')
            return

        # Iterate through each face_idx in the pano
        for face_idx in panos_info[pano]:
            face_data = panos_info[pano][face_idx]

            # Iterate through the masks, areas, and centroids for the current face_idx
            for mask, area, centroid in zip(face_data['masks'], face_data['areas'], face_data['centroids']):

                # Convert the mask to RLE
                rle = mask_util.encode(np.array(mask, order="F", dtype="uint8"))

                # Decode rle['counts'] to be able to save them in the json file
                rle['counts'] = rle['counts'].decode('utf-8')

                input_coco_format.append({
                    'pano_id': pano,
                    'face_idx': int(face_idx),
                    'area': int(area),
                    'centroid': [float(c) for c in centroid],  # Convert each element of centroid to float
                    'segmentation': rle
                })

    # Calculate max_threads based on CPU capacity
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent()
    max_threads = int((cpu_count * (1 - cpu_percent / 100)) * 0.4)

    # Use ThreadPoolExecutor to limit the number of concurrent threads
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        list(tqdm(executor.map(process_pano, panos), total=len(panos)))
        
    # Save input_coco_format
    with open(os.path.join(directory, f'input_coco_format_{args.threshold}.json'), 'w') as f:
        json.dump(input_coco_format, f)

def main(args):
    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f'Input directory: {args.input_dir}')
    print(f'Future input_coco_format name: input_coco_format_{args.threshold}.json')
    print(f'Segmentation directory: {args.seg_dir}')
    print(f'Output directory: {args.output_dir}')

    # 1. Resize masks to args.size x args.size
    resize_masks(args)
    
    # 2. Apply connected components + filtering
    panos_info = filter_masks(args)

    # (Optional) 2.1 Apply algorithm 2 (filtering by sidewalk bounding boxes)
    if args.algorithm == 2:
        panos_info, no_sidewalk_bbox_list = filter_masks_by_bboxes(args, panos_info)

        # Save the list of pano_ids and face_idxs that do not have a sidewalk mask in a .csv file
        with open(os.path.join(args.output_dir, f'no_sidewalk_masks_pano_list.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['pano_id', 'face'])
            for pano_id, face_name in no_sidewalk_bbox_list:
                writer.writerow([pano_id, face_name])

    # (Optional) 2.2 Apply algorithm 3 (filtering by sidewalk masks)
    if args.algorithm == 3:
        panos_info = filter_masks_by_masks(args, panos_info)

    # 3. Save the masks to a .json file
    save_masks(args, panos_info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='res/dataset', help='input directory')
    parser.add_argument('--size', type=int, default=2048, help='resize according to face size')
    parser.add_argument('--threshold', type=int, default='10')
    parser.add_argument('--output_dir', type=str, default='res/dataset', help='output directory')
    parser.add_argument('--algorithm', type=int, default=1, help='algorithm to use (1, 2, 3)')
    parser.add_argument('--seg_dir', type=str, default='res/seg', help='segmentation directory')

    args = parser.parse_args()

    main(args)