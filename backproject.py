'''Backproject the bounding boxes found in the left and right images to the equirectangular
panoramic image, preparing them as input for the triangulation algorithm.'''

import argparse
import os
import re
import sys
import json
import pycocotools.mask as mask_util
from pycocotools import mask
import imantics


from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt

import numpy as np
sys.path.append("lib")
import vrProjector as vrProjector

from tqdm import tqdm
    

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
    pano_mask = np.array(Image.open('res/equirectangular.png'))
    # Resize it to 2000x1000
    pano_mask = cv2.resize(pano_mask, (2000, 1000))

    # Define the RGB values for yellow
    yellow_rgb = np.array([255, 246, 0])
    # Define the RGB values for blue
    blue_rgb = np.array([0, 0, 255])
    # Convert it to HSV
    yellow_hsv = cv2.cvtColor(np.uint8([[yellow_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    blue_hsv = cv2.cvtColor(np.uint8([[blue_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

    # Threshold the image based on the HSV values
    yellow_lower = np.array([yellow_hsv[0] - 10, yellow_hsv[1], yellow_hsv[2]])
    yellow_upper = np.array([yellow_hsv[0] + 10, 255, 255])
    yellow_mask = cv2.inRange(cv2.cvtColor(pano_mask, cv2.COLOR_RGB2HSV), yellow_lower, yellow_upper)

    blue_lower = np.array([blue_hsv[0] - 10, blue_hsv[1], blue_hsv[2]])
    blue_upper = np.array([blue_hsv[0] + 10, 255, 255])
    blue_mask = cv2.inRange(cv2.cvtColor(pano_mask, cv2.COLOR_RGB2HSV), blue_lower, blue_upper)

    # Fill any small holes inside the masks
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    # Load the image
    img = cv2.imread(os.path.join(pano_path,pano) + '.png')

    # Convert all the image (img) but the pixels corresponding to the yellow_mask locations to black
    left_image_mask = img.copy()
    left_image_mask[yellow_mask != 255] = 0

    # Convert all the image (img) but the pixels corresponding to the blue_mask locations to black
    right_image_mask = img.copy()
    right_image_mask[blue_mask != 255] = 0

    return [left_image_mask, right_image_mask]

def backproject_masks(args, directory):
    # The idea is to create a black left image and just draw the bounding box on it. Then use
    # the back, front, right to project the bounding box on the equirectangular image.

    # If input_coco_format.json exists, load it
    if os.path.exists(os.path.join(directory, 'input_coco_format.json')):
        with open(os.path.join(directory, 'input_coco_format.json'), 'r') as f:
            input_coco_format = json.load(f)
    else:
        input_coco_format = []

    # The panos we want to process are the ones we have their masks
    panos_path = os.path.join(os.path.dirname(args.input_dir), 'masks_resized')
    panos = os.listdir(panos_path)

    # Print number of panos before filtering
    print('Number of panos before filtering:', len(panos))

    # Skip the ones we already processed
    panos = [pano for pano in panos if pano not in [pano['pano_id'] for pano in input_coco_format]]
    # Print number of panos after filtering
    print('Number of panos after filtering:', len(panos))

    # For testing, stop at 3k new panos
    panos = panos[:3000]

    # Make a for loop that goes through panoramas in data
    for pano in tqdm(panos):
        print('Processing panorama', pano)
        pano_path = os.path.join(args.input_dir, pano)
    
        #source = vrProjector.EquirectangularProjection()
        #source.loadImage(pano_path)
        #cb = vrProjector.CubemapProjection()
        #cb.initImages(512,512)
        #cb.reprojectToThis(source)

        # Make a directory for the backprojected panorama
        pano_path = os.path.join(directory, pano)
        if not os.path.exists(pano_path):
            os.makedirs(pano_path)


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
        back.save(back_path)

        #reprojected_pano_path = os.path.join(os.path.dirname(args.input_dir), 'reprojected', pano)

        #front_path = os.path.join(reprojected_pano_path, 'front.png')
        #back_path = os.path.join(reprojected_pano_path, 'back.png')

        masks_path = os.path.join(panos_path, pano)

        left_path = os.path.join(masks_path, 'left.png')
        right_path = os.path.join(masks_path, 'right.png')
    
        source = vrProjector.CubemapProjection()
        source.loadImages(front_path, right_path, back_path, left_path, top_path, bottom_path)
        eq = vrProjector.EquirectangularProjection()
        eq.initImage(2000,1000)
        eq.reprojectToThis(source)
        eq.saveImage(os.path.join(pano_path, f'{pano}.png'))

        # Delete the bottom,top,front,back images
        os.remove(top_path)
        os.remove(bottom_path)
        os.remove(front_path)
        os.remove(back_path)

        # Convert the masks to panorama sizes ones
        pano_masks = find_masks(pano_path, pano)

        #original_image_path = os.path.join(os.path.dirname(args.input_dir), 'reoriented', pano)

        # Prepare the data for the triangulation: make a .json file as a list of dictionaries,
        # where each dictionary contains the pano_id, the bounding box and the mask
        for pano_mask in pano_masks:

            # Threshold the pano_mask using np.where to make it binary
            # The original masks are made of confidence scores from the model, so they are not binary
            pano_mask = np.where(pano_mask > 0, 1, 0)

            # Convert the mask to RLE
            rles = mask_util.encode(np.array(pano_mask, order="F", dtype="uint8"))

            # Decode rle['counts'] to be able to save them in the json file
            for rle in rles:
                rle['counts'] = rle['counts'].decode('utf-8')


            input_coco_format.append({'pano_id': pano, 'segmentation': rle})

    # Save input_coco_format
    with open(os.path.join(directory, 'input_coco_format.json'), 'w') as f:
        json.dump(input_coco_format, f)

def main(args):
    # Replace everything that is not a character with an underscore in neighbourhood string, and make it lowercase
    args.neighbourhood = re.sub(r'[^a-zA-Z]', '_', args.neighbourhood).lower()

    args.input_dir = os.path.join(args.input_dir, args.neighbourhood)
    args.input_dir = args.input_dir + '_' + args.quality
    args.input_dir = os.path.join(args.input_dir, 'reoriented')

    # Create the output directory if it doesn't exist
    directory = os.path.join(os.path.dirname(args.input_dir), 'backprojected')
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    backproject_masks(args, directory)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--input_dir', type=str, default = 'res/dataset')
    parser.add_argument('--neighbourhood', type=str, default='osdorp')
    parser.add_argument('--quality', type=str, default='full')
    
    args = parser.parse_args()
    
    main(args)