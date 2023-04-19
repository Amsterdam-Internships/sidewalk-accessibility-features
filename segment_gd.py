import cv2
import json
import numpy as np
import argparse
import re
import os
import concurrent.futures
import psutil
from tqdm import tqdm

from PIL import Image

def segment_labels(input_image_path, masks_path, json_data, labels_to_blacken, output_image_path):

    masks = []
    labels = []

    original_image_path = input_image_path
    original_image = Image.open(original_image_path)
    
    # Loop through the json file and collect the labels. Each instance has a key "label" with the label name
    for instance in json_data:
        if instance["label"] not in labels and instance["label"] != "background":
            labels.append(instance["label"])

    # Check to remove labels not in labels_to_blacken
    labels = [label for label in labels if label in labels_to_blacken]

    for label in labels:
        mask_path = f'{label}_{masks_path}'
        mask = Image.open(mask_path)
        mask_image = mask.resize(original_image.size).convert('L')  # Convert to grayscale

        # Threshold the mask image to create a proper boolean mask
        threshold = 128
        mask_array = np.array(mask_image) > threshold
        masks.append(mask_array)
    
    # Overlay the mask on the original image and black the pixels that are in the mask
    for mask in masks:
        # Extend the mask_bool array to cover all three color channels (R, G, B)
        mask_bool = np.stack([mask, mask, mask], axis=-1)

        # Convert the original image to a numpy array
        original_image_array = np.array(original_image)

        # Black the pixels that are in the mask using np.where
        original_image_array = np.where(mask_bool, 0, original_image_array)

        # Convert the array back to an image
        original_image = Image.fromarray(original_image_array, 'RGB')

    # Save the image in the same folder as the original image
    original_image.save(output_image_path, "JPEG")

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

                segment_labels(input_image_path, masks_path, json_data, labels_to_blacken, output_image_path)
        else:
            # Case 1: Process files without directions
            input_image_path = os.path.join(subfolder, f"raw_image_{subfolder_name}.jpg")
            masks_path = os.path.join(subfolder, f"mask_{subfolder_name}.jpg")
            json_path = os.path.join(subfolder, f"mask_{subfolder_name}.json")
            output_image_path = os.path.join(output_folder, f"masked_{subfolder_name}.jpg")

            with open(json_path, "r") as json_file:
                json_data = json.load(json_file)

            segment_labels(input_image_path, masks_path, json_data, labels_to_blacken, output_image_path)

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