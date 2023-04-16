'''Backproject the bounding boxes found in the left and right images to the equirectangular
panoramic image, preparing them as input for the triangulation algorithm.'''
import sys
sys.path.append("lib")

import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import vrProjector as vrProjector
import argparse
import os
import re
import psutil
import concurrent.futures

import json
import pycocotools.mask as mask_util

'''def resize_masks(args, directory):
    #The function resizes the output masks from the model
    #to 512x512. We assume we use MOVE which outputs smaller masks.
    path = os.path.join(os.path.dirname(args.input_dir), 'masks_resized')
    directory = os.path.join(os.path.dirname(args.input_dir), 'masks')

    # Make a new directory to save the resized masks in
    if not os.path.exists(path):
        os.makedirs(path)

    # Import all the masks in res/dataset/centrum_west_small/masks and resize them to 512x512
    print(f'Resizing {len(os.listdir(directory))} masks...')
    for pano in tqdm(os.listdir(directory)):
        # Skip .DS_Store
        if pano == '.DS_Store':
            continue
        else:
            # Make a new directory in masks_resized
            pano_path = os.path.join(path, pano)
            if not os.path.exists(pano_path):
                os.makedirs(pano_path)

            # There are four masks in each pano folder: back, front, left, right. Resize them all to 512x512
            for mask in os.listdir(os.path.join(directory, pano)):
                mask_path = os.path.join(directory, pano, mask)
                mask = Image.open(mask_path)
                mask = mask.resize((512, 512))
                # Save them in res/dataset/centrum_west_small/masks_resized
                new_mask_path = os.path.join(path, pano)
                # If the name of the mask is 'left.png', save it as 'left.png' in the new directory, otherwise save it as 'right.png'
                if mask_path.endswith('left.png'):
                    mask.save(os.path.join(new_mask_path, 'left.png'))
                elif mask_path.endswith('right.png'):
                    mask.save(os.path.join(new_mask_path, 'right.png'))
                elif mask_path.endswith('front.png'):
                    mask.save(os.path.join(new_mask_path, 'front.png'))
                elif mask_path.endswith('back.png'):
                    mask.save(os.path.join(new_mask_path, 'back.png'))'''


'''https://stackoverflow.com/questions/49494337/encode-numpy-array-using-uncompressed-rle-for-coco-dataset'''
def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(binary_mask.ravel(order='F')):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle


def find_masks(pano_path, pano):
    # The idea is to use the image equirectangular.png in res/ and the colors HSV
    # (yellow: 57.9, 100, 100) and (blue: 240, 100, 100)
    # to define the left and right face cubemap images corresponding to the equirectangular image.
    # We do the same for the front and back, using the colors HSV (red: 0, 100, 100) and (green: 120, 100, 100)
    pano_mask = np.array(Image.open('res/equirectangular.png'))
    # Resize it to 2000x1000
    pano_mask = cv2.resize(pano_mask, (2000, 1000))

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

    # Load the image
    img = cv2.imread(os.path.join(pano_path, pano) + '.png')

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
    panos_masks_path = os.path.join(os.path.dirname(args.input_dir), 'masks')
    print(f'Looking for masks in {panos_masks_path}')
    panos = os.listdir(panos_masks_path)

    # Print number of panos before filtering
    print('Number of panos before filtering:', len(panos))

    # Skip the ones we already processed
    panos = [pano for pano in panos if pano not in [pano['pano_id']
                                                    for pano in input_coco_format]]
    # Print number of panos after filtering
    print('Number of panos after filtering:', len(panos))

    def process_pano(pano):
        print('Processing panorama', pano)

        # Make a directory for the backprojected panorama
        backproject_pano_path = os.path.join(directory, pano)
        if not os.path.exists(backproject_pano_path):
            os.makedirs(backproject_pano_path)

        '''
        # Make bottom,top,front,back 512x512 black images
        bottom = Image.new('RGB', (512, 512), (0, 0, 0))
        top = Image.new('RGB', (512, 512), (0, 0, 0))
        front = Image.new('RGB', (512, 512), (0, 0, 0))
        back = Image.new('RGB', (512, 512), (0, 0, 0))
        # Save them
        top_path = os.path.join(pano_path, 'top.png')
        bottom_path = os.path.join(pano_path, 'bottom.png')
        front_path = os.path.join(pano_path, 'front.png')
        back_path = os.path.join(pano_path, 'back.png')

        bottom.save(bottom_path)
        top.save(top_path)
        front.save(front_path)
        back.save(back_path)'''

        masks_path = os.path.join(panos_masks_path, pano)

        # Load the left, right, front, back masks path.
        # If they don't exist, make a 512x512 black image and save it in the folder.
        # This is the case of top and bottom masks
        left_path = os.path.join(masks_path, 'left.png')
        right_path = os.path.join(masks_path, 'right.png')
        front_path = os.path.join(masks_path, 'front.png')
        back_path = os.path.join(masks_path, 'back.png')
        top_path = os.path.join(masks_path, 'top.png')
        bottom_path = os.path.join(masks_path, 'bottom.png')
        '''if not os.path.exists(left_path):
            left = Image.new('RGB', (512, 512), (0, 0, 0))
            left.save(left_path)
        if not os.path.exists(right_path):
            right = Image.new('RGB', (512, 512), (0, 0, 0))
            right.save(right_path)
        if not os.path.exists(front_path):
            front = Image.new('RGB', (512, 512), (0, 0, 0))
            front.save(front_path)
        if not os.path.exists(back_path):
            back = Image.new('RGB', (512, 512), (0, 0, 0))
            back.save(back_path)'''
        if not os.path.exists(top_path):
            top = Image.new('RGB', (512, 512), (0, 0, 0))
            top.save(top_path)
        if not os.path.exists(bottom_path):
            bottom = Image.new('RGB', (512, 512), (0, 0, 0))
            bottom.save(bottom_path)

        source = vrProjector.CubemapProjection()
        source.loadImages(front_path, right_path, back_path,
                          left_path, top_path, bottom_path)
        eq = vrProjector.EquirectangularProjection()
        eq.initImage(2000, 1000)
        eq.reprojectToThis(source)
        eq.saveImage(os.path.join(backproject_pano_path, f'{pano}.png'))

        # Delete the bottom,top,front,back images
        #os.remove(top_path)
        #os.remove(bottom_path)
        #os.remove(left_path)
        #os.remove(right_path)
        #os.remove(front_path)
        #os.remove(back_path)

        # Convert the masks to panorama sizes ones
        pano_masks = find_masks(masks_path, pano)

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

    if args.ps:
        args.input_dir = os.path.join(args.input_dir, 'reoriented')
    else:
        args.input_dir = os.path.join(args.input_dir, args.neighbourhood)
        args.input_dir = args.input_dir + '_' + args.quality
        args.input_dir = os.path.join(args.input_dir, 'reoriented')

    # Create the output directory if it doesn't exist
    directory = os.path.join(os.path.dirname(args.input_dir), 'backprojected')

    if not os.path.exists(directory):
        os.makedirs(directory)

    #resize_masks(args, directory)
    backproject_masks(args, directory)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='res/dataset')
    parser.add_argument('--neighbourhood', type=str, default='osdorp')
    parser.add_argument('--quality', type=str, default='full')
    parser.add_argument('--ps', type=bool, default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    main(args)
