# -*- coding: utf-8 -*-
'''
Extract the part of the image corresponding to the mask obtained from
an object discovery model.
Input: list of masks, list of reprojected panoramas
Output: list of cropped images

The script can also be used to overlay the mask on top of the original image.
To do so, set the overlay flag to True.
TODO: Make a separate script for this functionality.

@author: Andrea Lombardo
'''

import psutil

import re
import os
import time
import argparse
import json

import numpy as np
from PIL import Image
from tqdm import tqdm   
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, find_contours
from skimage.filters import threshold_otsu

'''Parts of the code are taken from 
https://github.com/nikhilroxtomar/Semantic-Segmentation-Mask-to-Bounding-Box/blob/main/mask_to_bbox.py'''

""" Convert a mask to border image """
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    threshold = threshold_otsu(mask)

    contours = find_contours(mask, threshold)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

""" Mask to bounding boxes """
def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes

def parse_mask(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    
    return mask

def contours(mask):

    # Make sure the image has the correct data type (8-bit unsigned integer)
    #mask_8bit = mask.copy()
    #mask_8bit = mask_8bit.astype(np.uint8)

    # Find contours of the mask
    #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area, in descending order
    #contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find the bounding rectangle of the two biggest contours
    #x, y, width, height = cv2.boundingRect(contours[0])
    #x2, y2, width2, height2 = cv2.boundingRect(contours[1])

    # Detect bounding boxes
    bboxes = mask_to_bbox(mask)

    # Convert bboxes to numpy array
    bboxes = np.array(bboxes)

    '''# Visualize the bounding boxes
    for bbox in bboxes:
        x, y, width, height = bbox
        cv2.rectangle(mask, (x, y), (width, height), (255, 255, 255), 2)

    plt.imshow(mask, cmap='gray')
    plt.show()'''

    # Sort the bounding boxes by area, in descending order
    bboxes = sorted(bboxes, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), reverse=True)

    # If bboxes has less than 2 elements, consider only the first one
    if len(bboxes) < 2:
        x = bboxes[0][0]
        y = bboxes[0][1]
        width = bboxes[0][2] - x
        height = bboxes[0][3] - y
    else:
        # Take the first two bounding boxes and find the one enclosing them
        x0, y0, x1, y1 = bboxes[0]
        x2, y2, x3, y3 = bboxes[1]

        # Find the coordinates of the top left and bottom right points of the enclosing box
        x = min(x0, x2)
        y = min(y0, y2)
        width = max(x1, x3) - x
        height = max(y1, y3) - y

    cv2.rectangle(mask, (x, y), (x + width, y + height), (255, 255, 255), 2)

    # Calculate the coordinates of the four points of the bounding box
    tl = (x, y)
    tr = (x + width, y)
    br = (x + width, y + height)
    bl = (x, y + height)

    # Output the 4 points of the box in 2D coordinates
    bbox = [tl, tr, br, bl]

    # Convert the points to int to serialize them later
    for i in range(len(bbox)):
        bbox[i] = (int(bbox[i][0]), int(bbox[i][1]))

    return bbox

def extract(args, overlay):
    # The folder contains a list of folders, each one containing the masks for back, front, left and right images
    # We are interested in just the left and right ones
    # Ignore the .DS_Store file if it exists
    # We want to do this for both the original folder and the mask folder, together
    imgs = ['/left.png', '/right.png']
    neighbourhood = args.neighbourhood + '_' + args.quality
    masks_folder = os.path.join(os.path.dirname(args.input_dir), 'masks')

    # Define a dictionary which will have a pano_id key, its value is panorama_folder,
    # The left and right images are linked to that value and each image has a bounding box
    dict_bbox = {}

    # Make a .csv file to store the bounds of the mask. The first row is 'pano_id', the second is 'img', the third is 'coords'
    for panorama_folder in tqdm(os.listdir(masks_folder)):
        if panorama_folder != '.DS_Store':
            # Make sub-folders for each panorama
            if (overlay):
                os.makedirs('res/dataset/' + neighbourhood + '/masked/' + panorama_folder, exist_ok=True)
            else:
                os.makedirs('res/dataset/' + neighbourhood + '/extracted/' + panorama_folder, exist_ok=True)
            for img in imgs:
                # Load list of files in the folder
                # File is in format MOVE_neighbourhood_full_masks/panorama_folder/left.png
                file_path = os.path.join(masks_folder, panorama_folder) + img
                # Load list of files in the folder of original images
                # File is in format neighbourhood_full_reprojected/panorama_folder/left.png
                file_path_original = os.path.join(args.input_dir, panorama_folder) + img
                # Load the original image
                original_image = cv2.imread(file_path_original)

                # Convert the colors
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                # Load the mask
                mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                # Apply Gaussian blur to the mask to smooth it out
                kernel_size = (3,3)
                mask = cv2.GaussianBlur(mask, kernel_size, 0)

                # Resize the mask to the size of the original image
                mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
                # Normalize the mask
                normalized_mask = (mask - mask.min()) / (mask.max() - mask.min())

                # Print the panorama id
                print('Panorama id: ' + panorama_folder)
                print('Image is: ' + img)

                # Transform the mask into a bounding box and save the coordinates into a .json file
                bbox = contours(normalized_mask)

                # Strip /left.png or /right.png from the slash and the extension
                img_name = img[1:-4]

                # Add the panorama id, the image name and the coordinates to the dictionary keys
                # in such a way that the key will contain multiple panorama ids, image names and coordinates
                dict_bbox.setdefault(panorama_folder, {}).setdefault(img_name, []).append(bbox)

                print(dict_bbox)

                if (overlay):
                    # Create a new figure with the same dimension as the original image
                    dpi = 100
                    figsize = original_image.shape[1]/dpi, original_image.shape[0]/dpi
                    fig = plt.figure(figsize=figsize)
                    # Create one axis
                    ax = fig.add_axes([0, 0, 1, 1])
                    ax.axis('off')
                    # Overlay the mask on top of the original image with transparency, cmap = grayscale
                    ax.imshow(original_image)
                    ax.imshow(normalized_mask, cmap='gray', alpha=0.4)
                    # Save the image, make sure to remove the border
                    save_path = '/home/andrealombardo/Documents/GitHub/sidewalk-accessibility-features/res/dataset/' + neighbourhood + '/masked/' + panorama_folder + img
                    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
                else:
                    # Extract only the part of the image that corresponds to the mask
                    result = np.zeros(original_image.shape, np.uint8)
                    result[normalized_mask != 0] = original_image[normalized_mask != 0]
                    # Save the image in MOVE_centrum_west_full, making a folder for each panorama, and saving the left and right images in the same folder
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    save_path = '/home/andrealombardo/Documents/GitHub/sidewalk-accessibility-features/res/dataset/' + neighbourhood + '/extracted/' + panorama_folder + img
                    cv2.imwrite(save_path, result)

                # Save the dictionary into a .json file
                with open('res/dataset/' + neighbourhood + '/bboxes.json', 'w') as f:
                    json.dump(dict_bbox, f)

def main(args):

    # Replace everything that is not a character with an underscore in neighbourhood string, and make it lowercase
    args.neighbourhood = re.sub(r'[^a-zA-Z]', '_', args.neighbourhood).lower()

    args.input_dir = os.path.join(args.input_dir, args.neighbourhood)
    args.input_dir = args.input_dir + '_' + args.quality
    args.input_dir = os.path.join(args.input_dir, 'reprojected')

    # Create the output directory if it doesn't exist
    # Take args.input_dir, strip the last part of the path and add the function name
    if (args.overlay):
        directory = os.path.join(os.path.dirname(args.input_dir), 'masked')
    else:
        directory = os.path.join(os.path.dirname(args.input_dir), 'extracted')

    if not os.path.exists(directory):
                os.makedirs(directory)

    extract(args, args.overlay)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--input_dir', type=str, default = 'res/dataset')
    parser.add_argument('--neighbourhood', type=str, default='osdorp')
    parser.add_argument('--quality', type=str, default='full')
    parser.add_argument('--size', type=int, default = 512)
    parser.add_argument('--overlay', type=bool, default = False)
    
    args = parser.parse_args()
    
    main(args)