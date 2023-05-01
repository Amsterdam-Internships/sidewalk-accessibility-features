'''Script that runs through the faces of the pano dataset, takes the Grounded-SAM results of sidewalk masks
and blackens everything else in the image. We hope that this will help the UOD model to find the right objects.'''
import os
import cv2
import numpy as np
import json
import argparse
from tqdm import tqdm
from scipy import ndimage

def blacken_above_max_y(mask, image, buffer=10):
    assert mask.shape == image.shape[:2], "Mask and image must have the same dimensions"
    
    # Compute the bounding box of the mask using ndimage
    labeled_mask, num_labels = ndimage.label(mask)
    if num_labels == 0:
        return image
    
    bounding_boxes = ndimage.find_objects(labeled_mask)
    max_y = max([bb[0].stop for bb in bounding_boxes if bb is not None])

    # Blacken everything above max_y + buffer in the image
    image[:max_y + buffer] = 0
    return image

def blacken_panos(args):
    # Traverse the pano folders
    for pano in tqdm(os.listdir(args.input_dir)):
        panos_path = os.path.join(args.input_dir, pano)
        faces = os.listdir(panos_path)

        # Create the pano folder in the output directory if it doesn't exist
        if not os.path.exists(os.path.join(args.output_dir, pano)):
            os.makedirs(os.path.join(args.output_dir, pano))

        # Check if we already went through this pano. In order to do so, check if the number of files inside
        # args.input_dir/pano is the same as the number of files inside args.output_dir/pano
        if len(os.listdir(panos_path)) == len(os.listdir(os.path.join(args.output_dir, pano))):
            print(f'{pano}: pano already processed. Skipping...')
            continue

        # Traverse the faces
        for face_file in faces:
            # Load the face
            face = cv2.imread(os.path.join(panos_path, face_file))
            # Load the mask. First, check if the mask exists. If it doesn't, skip the pano
            if not os.path.exists(os.path.join(args.masks_dir, pano, f'mask_{face_file}')):
                print(f'Mask not found for {face_file}. Skipping...')
                continue
            mask = cv2.imread(os.path.join(args.masks_dir, pano, f'mask_{face_file}'), cv2.IMREAD_GRAYSCALE)
            # Blacken everything above the max y of the mask + buffer
            b_face = blacken_above_max_y(mask, face, args.buffer)
            # Save the face
            cv2.imwrite(os.path.join(args.output_dir, pano, face_file), b_face)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='res/dataset', help='input directory')
    parser.add_argument('--masks_dir', type=str, default='res/grounded-sam', help='masks directory')
    parser.add_argument('--output_dir', type=str, default='res/dataset', help='output directory')
    parser.add_argument('--buffer', type=int, default=10, help='buffer for blackening (in pixels)')

    args = parser.parse_args()

    main(args)