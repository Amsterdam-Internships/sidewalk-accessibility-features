import cv2
import json
import numpy as np
import argparse
import re
import os
import concurrent.futures
import psutil
from tqdm import tqdm

def find_label_indices(json_data, target_labels):
    indices = []
    for i, item in enumerate(json_data):
        label = re.sub(r'\W+', '', item['label'])
        if label in target_labels:
            indices.append(i)
    return indices

def apply_mask(pano_image, mask_image, json_data, target_label):
    target_label_indices = find_label_indices(json_data, target_label)
    masked_pano = pano_image.copy()

    for idx in target_label_indices:
        mask_value = json_data[idx]["value"]
        blackened_pixels = (mask_image == mask_value)
        masked_pano[blackened_pixels] = 0

    return masked_pano

def blacken_labels(input_image_path, masks_path, json_data, labels_to_blacken, output_image_path):
    # Load the panoramic image and create a copy
    image = cv2.imread(input_image_path)
    image_copy = np.copy(image)
    

    # Load the masks as an image
    # Mask size: 2325 × 1162
    masks = cv2.imread(masks_path, cv2.IMREAD_GRAYSCALE)

    # Resize the masks image to match the input image dimensions (INTER_AREA should work best for downsampling)
    masks = cv2.resize(masks, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

    masked_pano = apply_mask(image_copy, masks, json_data, labels_to_blacken)

    # Save the modified image
    cv2.imwrite(output_image_path, masked_pano)

def main(args):
    # Replace everything that is not a character with an underscore in neighbourhood string, and make it lowercase
    args.neighbourhood = re.sub(r'[^a-zA-Z]', '_', args.neighbourhood).lower()

    args.labels = args.labels.split('.')
    labels_to_blacken = args.labels

    folder_name = 'ground_seg_outputs'

    if args.ps:
        args.input_dir = os.path.join(args.input_dir, folder_name)
        
    else:
        args.input_dir = os.path.join(args.input_dir, args.neighbourhood)
        args.input_dir = args.input_dir + '_' + args.quality
        args.input_dir = os.path.join(args.input_dir, folder_name)

    def process_subfolder(subfolder, output_folder, labels_to_blacken):
        # Save subfolder name
        subfolder_name = os.path.basename(subfolder)

        # Check if there are separate files for each direction
        has_directions = any(any(direction in f for direction in \
                             ["_front", "_back", "_right", "_left"]) for f in os.listdir(subfolder))

        # Process each subfolder depending on the case
        if has_directions:
            # Case 2: Process files with directions
            directions = ["front", "back", "right", "left"]
            # Save all the files in a folder with subfolder_name as name
            output_folder = os.path.join(output_folder, subfolder_name)
            for direction in directions:
                input_image_path = os.path.join(subfolder, f"raw_image_{direction}.jpg")
                masks_path = os.path.join(subfolder, f"mask_{direction}.jpg")
                json_path = os.path.join(subfolder, f"mask_{direction}.json")
                output_image_path = os.path.join(output_folder, f"{direction}.png")

                with open(json_path, "r") as json_file:
                    json_data = json.load(json_file)

                blacken_labels(input_image_path, masks_path, json_data, labels_to_blacken, output_image_path)
        else:
            # Case 1: Process files without directions
            input_image_path = os.path.join(subfolder, f"raw_image_{subfolder_name}.jpg")
            masks_path = os.path.join(subfolder, f"mask_{subfolder_name}.jpg")
            json_path = os.path.join(subfolder, f"mask_{subfolder_name}.json")
            output_image_path = os.path.join(output_folder, f"{subfolder_name}.jpg")

            with open(json_path, "r") as json_file:
                json_data = json.load(json_file)

            blacken_labels(input_image_path, masks_path, json_data, labels_to_blacken, output_image_path)

    # Define the input and output folders
    input_folder = args.input_dir
    
    root_folder = os.path.dirname(input_folder)
    output_folder = os.path.join(root_folder, 'reoriented_seg')

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all subfolders in the input folder
    subfolders = [os.path.join(input_folder, d) for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]

    # Calculate max_threads based on CPU capacity
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent()
    max_threads = int((cpu_count * (1 - cpu_percent / 100)) * 0.5)
    print(f'Using {max_threads} threads')

    # Use ThreadPoolExecutor to limit the number of concurrent threads
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        list(tqdm(executor.map(lambda subfolder: process_subfolder(subfolder, output_folder, labels_to_blacken), subfolders), total=len(subfolders)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='res/dataset')
    parser.add_argument('--neighbourhood', type=str, default='osdorp')
    parser.add_argument('--quality', type=str, default='full')
    parser.add_argument('--ps', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--labels', type=str, default='sky.building.car')

    args = parser.parse_args()

    main(args)