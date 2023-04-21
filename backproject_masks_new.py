'''Backproject the bounding boxes found in the left and right images to the equirectangular
panoramic image, preparing them as input for the triangulation algorithm.'''
import sys
sys.path.append("lib")

import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from lib.py360convert import py360convert
import argparse
import os
import re
import psutil
import concurrent.futures

import json
import pycocotools.mask as mask_util

def find_masks(pano_path, pano):
    # The idea is to use the image equirectangular.png in res/ and the colors HSV
    # (yellow: 57.9, 100, 100) and (blue: 240, 100, 100)
    # to define the left and right face cubemap images corresponding to the equirectangular image.
    # We do the same for the front and back, using the colors HSV (red: 0, 100, 100) and (green: 120, 100, 100)
    pano_mask = np.array(Image.open('res/equirectangular.png'))
    # Resize it to 2000x1000
    pano_mask = cv2.resize(pano_mask, (2000, 1000))

    # Load the image
    img = cv2.imread(os.path.join(pano_path, pano) + '.png')

    # Define the RGB values for yellow
    yellow_rgb = np.array([255, 246, 0])
    # Define the RGB values for blue
    blue_rgb = np.array([0, 0, 255])
    # Define the RGB values for red
    red_rgb = np.array([255, 0, 0])
    # Define the RGB values for green
    green_rgb = np.array([0, 255, 0])
    
    # Convert it to HSV
    yellow_hsv = cv2.cvtColor(
        np.uint8([[yellow_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    blue_hsv = cv2.cvtColor(np.uint8([[blue_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    red_hsv = cv2.cvtColor(np.uint8([[red_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    green_hsv = cv2.cvtColor(np.uint8([[green_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

    # Threshold the image based on the HSV values
    yellow_lower = np.array([yellow_hsv[0] - 10, yellow_hsv[1], yellow_hsv[2]])
    yellow_upper = np.array([yellow_hsv[0] + 10, 255, 255])
    yellow_mask = cv2.inRange(cv2.cvtColor(
        pano_mask, cv2.COLOR_RGB2HSV), yellow_lower, yellow_upper)

    blue_lower = np.array([blue_hsv[0] - 10, blue_hsv[1], blue_hsv[2]])
    blue_upper = np.array([blue_hsv[0] + 10, 255, 255])
    blue_mask = cv2.inRange(cv2.cvtColor(
        pano_mask, cv2.COLOR_RGB2HSV), blue_lower, blue_upper)
    
    red_lower = np.array([red_hsv[0] - 10, red_hsv[1], red_hsv[2]])
    red_upper = np.array([red_hsv[0] + 10, 255, 255])
    red_mask = cv2.inRange(cv2.cvtColor(
        pano_mask, cv2.COLOR_RGB2HSV), red_lower, red_upper)
    
    green_lower = np.array([green_hsv[0] - 10, green_hsv[1], green_hsv[2]])
    green_upper = np.array([green_hsv[0] + 10, 255, 255])
    green_mask = cv2.inRange(cv2.cvtColor(
        pano_mask, cv2.COLOR_RGB2HSV), green_lower, green_upper)

    # Fill any small holes inside the masks
    yellow_mask = cv2.morphologyEx(
        yellow_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    blue_mask = cv2.morphologyEx(
        blue_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    red_mask = cv2.morphologyEx(
        red_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    green_mask = cv2.morphologyEx(
        green_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))



    # Convert all the image (img) but the pixels corresponding to the yellow_mask locations to black
    left_image_mask = img.copy()
    left_image_mask[yellow_mask != 255] = 0

    # Convert all the image (img) but the pixels corresponding to the blue_mask locations to black
    right_image_mask = img.copy()
    right_image_mask[blue_mask != 255] = 0

    # Convert all the image (img) but the pixels corresponding to the red_mask locations to black
    front_image_mask = img.copy()
    front_image_mask[red_mask != 255] = 0

    # Convert all the image (img) but the pixels corresponding to the green_mask locations to black
    back_image_mask = img.copy()
    back_image_mask[green_mask != 255] = 0

    return [left_image_mask, right_image_mask, front_image_mask, back_image_mask]


def backproject_masks(args, directory):
    # If input_coco_format.json exists, load it
    if os.path.exists(os.path.join(directory, 'input_coco_format.json')):
        with open(os.path.join(directory, 'input_coco_format.json'), 'r') as f:
            input_coco_format = json.load(f)
    else:
        input_coco_format = []

    # The panos we want to process are the ones we have their masks
    panos_masks_path = os.path.join(args.input_dir, args.masks_dir)
    print(f'Looking for masks in {panos_masks_path}')
    panos = os.listdir(panos_masks_path)

    # Print number of panos before filtering
    print('Number of panos before filtering:', len(panos))

    # Skip the ones we already processed
    panos = [pano for pano in panos if pano not in [pano['pano_id']
                                                    for pano in input_coco_format]]
    # Print number of panos after filtering
    print('Number of panos after filtering:', len(panos))

    # Test evaluation function
    #panos = panos[:100]
    #print(f'Testing the following panos: {panos}')

    def process_pano(pano):
        print('Processing panorama', pano)

        masks_path = os.path.join(panos_masks_path, pano)

        # Check if there are less then 4 masks in the folder.
        # If so, skip this panorama and save pano in a new .csv called 'not_backprojected.csv'
        masks = os.listdir(masks_path)
        if len(masks) < 4:
            print('Less than 4 masks found, skipping panorama')
            with open(os.path.join(directory, f'not_{args.backprojected_dir}.csv'), 'a') as f:
                f.write(f'{pano}\n')
            return
        
        # Make a directory for the backprojected panorama
        backproject_pano_path = os.path.join(directory, pano)
        if not os.path.exists(backproject_pano_path):
            os.makedirs(backproject_pano_path)


        # Load the left, right, front, back masks path.
        # If they don't exist, make a 512x512 black image and save it in the folder.
        # This is the case of top and bottom masks
        left_path = os.path.join(masks_path, 'left.png')
        right_path = os.path.join(masks_path, 'right.png')
        front_path = os.path.join(masks_path, 'front.png')
        back_path = os.path.join(masks_path, 'back.png')
        top_path = os.path.join(masks_path, 'top.png')
        bottom_path = os.path.join(masks_path, 'bottom.png')

        if not os.path.exists(top_path):
            top = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.imwrite(top_path, top)
        if not os.path.exists(bottom_path):
            bottom = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.imwrite(bottom_path, bottom)

        # Load the masks
        left = cv2.imread(left_path)
        right = cv2.imread(right_path)
        front = cv2.imread(front_path)
        back = cv2.imread(back_path)
        top = cv2.imread(top_path)
        bottom = cv2.imread(bottom_path)

        # Project from cubemap to equirectangular
        equirectangular = py360convert.c2e([front, cv2.flip(right, 1), cv2.flip(back, 1), left, cv2.flip(top, 0), bottom], \
                                        w=2000, h=1000, mode='bilinear', cube_format='list')
        cv2.imwrite(os.path.join(backproject_pano_path, f'{pano}.png'), equirectangular)

        # Convert the masks to panorama sizes ones
        pano_masks = find_masks(backproject_pano_path, pano)

        # original_image_path = os.path.join(os.path.dirname(args.input_dir), 'reoriented', pano)

        # Prepare the data for the triangulation: make a .json file as a list of dictionaries,
        # where each dictionary contains the pano_id, the bounding box and the mask
        for pano_mask in pano_masks:

            # Threshold the pano_mask using np.where to make it binary
            # The original masks are made of confidence scores from the model, so they are not binary
            pano_mask = np.where(pano_mask > 0, 1, 0)

            # Convert the mask to RLE
            rles = mask_util.encode(
                np.array(pano_mask, order="F", dtype="uint8"))

            # Decode rle['counts'] to be able to save them in the json file
            for rle in rles:
                rle['counts'] = rle['counts'].decode('utf-8')

            input_coco_format.append({'pano_id': pano, 'segmentation': rle})

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
    # Replace everything that is not a character with an underscore in neighbourhood string, and make it lowercase
    args.neighbourhood = re.sub(r'[^a-zA-Z]', '_', args.neighbourhood).lower()

    if not args.ps:
        args.input_dir = os.path.join(args.input_dir, args.neighbourhood)
        args.input_dir = args.input_dir + '_' + args.quality
    
    # Create the output directory if it doesn't exist
    directory = os.path.join(args.input_dir, args.backprojected_dir)

    if not os.path.exists(directory):
        os.makedirs(directory)

    print('Input directory: ', args.input_dir)
    print('Masks directory: ', args.masks_dir)
    print('Output directory: ', directory)

    #resize_masks(args, directory)
    backproject_masks(args, directory)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='res/dataset')
    parser.add_argument('--neighbourhood', type=str, default='osdorp')
    parser.add_argument('--quality', type=str, default='full')
    parser.add_argument('--ps', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--backprojected_dir', type=str, default='backprojected')
    parser.add_argument('--masks_dir', type=str, default='masks')

    args = parser.parse_args()

    main(args)
