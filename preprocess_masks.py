import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import json
import pycocotools.mask as mask_util
import pandas as pd
import concurrent.futures
import psutil

def reverse_map_faces(face):
    # Function that maps the face name to the corresponding index
    # Mapping: 'front' -> 0, 'right' -> 1, 'back' -> 2, 'left' -> 3, 'top' -> 4, 'bottom' -> 5
    face_mapping = {'front': 0, 'right': 1, 'back': 2, 'left': 3, 'top': 4, 'bottom': 5}
    return face_mapping.get(face)

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
            # Load the mask
            mask = cv2.imread(os.path.join(pano_masks_path, pano_mask), cv2.IMREAD_GRAYSCALE)

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

    # For each pano folder in args.output_dir, go through the masks, apply connected component analysis and filter the masks
    for pano in tqdm(os.listdir(directory)):
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
                    filtered_mask[output == i] = 255
                    masks.append(filtered_mask)
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

    print('Done!')
    return panos_info

def save_masks(args, panos_info):
    directory = args.output_dir

    # If input_coco_format.json exists, load it
    if os.path.exists(os.path.join(directory, 'input_coco_format.json')):
        with open(os.path.join(directory, 'input_coco_format.json'), 'r') as f:
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
    with open(os.path.join(directory, 'input_coco_format.json'), 'w') as f:
        json.dump(input_coco_format, f)

def main(args):
    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Input directory: ', args.input_dir)
    print('Output directory: ', args.output_dir)

    # 1. Resize masks to args.size x args.size
    resize_masks(args)
    
    # 2. Apply connected components + filtering
    panos_info = filter_masks(args)

    # 3. Save the masks to a .json file
    save_masks(args, panos_info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='res/dataset', help='input directory')
    parser.add_argument('--size', type=int, default=2048, help='resize according to face size')
    parser.add_argument('--threshold', type=int, default='10')
    parser.add_argument('--output_dir', type=str, default='res/dataset', help='output directory')

    args = parser.parse_args()

    main(args)