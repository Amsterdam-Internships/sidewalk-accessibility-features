'''Script that runs through the faces of the pano dataset, takes the Grounded-SAM results of sidewalk masks
and blackens everything else in the image. We hope that this will help the UOD model to find the right objects.'''
import os
import cv2
import numpy as np
import json
import argparse
from tqdm import tqdm
import concurrent.futures
import psutil

def blacken_above_max_y(image, mask, buffer=10, json_file=None):

    if json_file is None:
        return image

    # Load the corresponding .json file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Load all the bounding boxes coordinates, background excluded
    bounding_boxes = [item["box"] for item in data if item["label"] != "background"]

    # Rescale the bounding boxes to image.shape
    height, width = image.shape[:2]
    source_height, source_width = mask.shape[:2]
    rescaled_bounding_boxes = []
    for bb in bounding_boxes:
        rescaled_bb = [
            bb[0] * (width / source_width),
            bb[1] * (height / source_height),
            bb[2] * (width / source_width),
            bb[3] * (height / source_height),
        ]
        rescaled_bounding_boxes.append(rescaled_bb)

    # Find the max_y among bounding boxes
    max_y = max([bb[3] for bb in rescaled_bounding_boxes])

    # Blacken everything above max_y + buffer in the image
    image[:int(max_y) + buffer] = 0
    return image

def blacken_panos(args):
    # Traverse the pano folders
    panos = os.listdir(args.input_dir)

    def process_pano(pano):
        panos_path = os.path.join(args.input_dir, pano)
        faces = os.listdir(panos_path)

        # Create the pano folder in the output directory if it doesn't exist
        if not os.path.exists(os.path.join(args.output_dir, pano)):
            os.makedirs(os.path.join(args.output_dir, pano))

        # Check if we already went through this pano. In order to do so, check if the number of files inside
        # args.input_dir/pano is the same as the number of files inside args.output_dir/pano
        if len(os.listdir(panos_path)) == len(os.listdir(os.path.join(args.output_dir, pano))):
            print(f'{pano}: pano already processed. Skipping...')
            return

        # Check if the masks directory contains the masks for this pano. If it doesn't, skip the pano
        if not os.path.exists(os.path.join(args.masks_dir, pano)):
            print(f'{pano}: masks not found. Skipping...')
            return

        # Traverse the faces
        for face_file in faces:
    
            # Load the face
            face_name = face_file.split('.')[0]
            face = cv2.imread(os.path.join(panos_path, face_file))
            # Load the mask containing the bounding boxes. First, check if the file exists. 
            # If it doesn't, skip the pano
            mask_file = f'mask_{face_name}.jpg'
            mask_path = os.path.join(args.masks_dir, pano, mask_file)
            if not os.path.exists(mask_path):
                print(f'Mask not found for {face_name}. Skipping...')
                continue
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Load the json file containing the bounding boxes
            json_file = f'mask_{face_name}.json'
            json_path = os.path.join(args.masks_dir, pano, json_file)
            if not os.path.exists(json_path):
                print(f'Json file not found for {face_name}. Skipping...')
                continue
            # Blacken everything above the max y of the mask + buffer
            b_face = blacken_above_max_y(face, mask, args.buffer, json_path)
            # Save the face
            cv2.imwrite(os.path.join(args.output_dir, pano, face_file), b_face)

    # Calculate max_threads based on CPU capacity
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent()
    max_threads = int((cpu_count * (1 - cpu_percent / 100)) * 0.5)

    # Use ThreadPoolExecutor to limit the number of concurrent threads
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        list(tqdm(executor.map(process_pano, panos), total=len(panos)))
        

def main(args):
    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f'Input directory: {args.input_dir}')
    print(f'Masks directory: {args.masks_dir}')
    print(f'Output directory: {args.output_dir}')
    print(f'Buffer: {args.buffer}')

    # 1. Resize masks to args.size x args.size
    blacken_panos(args)

    # 2. Remove empty folders
    for pano in os.listdir(args.output_dir):
        if len(os.listdir(os.path.join(args.output_dir, pano))) == 0:
            os.rmdir(os.path.join(args.output_dir, pano))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='res/dataset', help='input directory')
    parser.add_argument('--masks_dir', type=str, default='res/grounded-sam', help='masks directory')
    parser.add_argument('--output_dir', type=str, default='res/dataset', help='output directory')
    parser.add_argument('--buffer', type=int, default=10, help='buffer for blackening (in pixels)')

    args = parser.parse_args()

    main(args)