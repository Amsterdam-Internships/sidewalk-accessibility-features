# -*- coding: utf-8 -*-
'''
Reproject images from equirectangular to cubemap.
Credits to Tim Alpherts for the split function.

@author: Andrea Lombardo
'''

import psutil

import re
import os
import concurrent.futures
import lib.vrProjector as vrProjector
import time
import argparse
import pandas as pd
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm     

def split(args, img_path, directory):

    print(f'Processing image...')
    img = os.path.join(args.input_dir, img_path)

    # VrProjector the images 
    size = args.size
    eq = vrProjector.EquirectangularProjection()
    try:
        eq.loadImage(img)
    except FileNotFoundError:
        print(f'File {img} not found')
        return
    cb = vrProjector.CubemapProjection()
    cb.initImages(size,size)
    cb.reprojectToThis(eq)

    # retrieve front back left right
    front = Image.fromarray(np.uint8(cb.front))
    right = Image.fromarray(np.uint8(cb.right))
    back = Image.fromarray(np.uint8(cb.back))
    left = Image.fromarray(np.uint8(cb.left))

    # make directory, with panoid as name, to save them in
    folder = img_path.split('.')[0]
    directory = os.path.join(directory, folder)
    if not os.path.exists(directory):
            os.makedirs(directory)
    # save them in that dir
    front.save(os.path.join(directory, 'front.png'))
    right.save(os.path.join(directory, 'right.png'))
    back.save(os.path.join(directory, 'back.png'))
    left.save(os.path.join(directory, 'left.png'))

def reproject_panos(args, root_dir):
    # Define list of images in the input directory, skip other formats
    img_list = [img for img in os.listdir(args.input_dir) if img.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]
    print('Number of reoriented panos: {}'.format(len(img_list)))

    # Define output directory
    directory = os.path.join(os.path.dirname(args.input_dir), args.output_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Copy img_list and remove extensions from img_list_copy
    img_list_copy = img_list.copy()
    for i in range(len(img_list_copy)):
        img_list_copy[i] = img_list_copy[i].split('.')[0]

    # Remove reprojected images that are not in img_list
    for entry in os.scandir(directory):
        if entry.is_dir():
            # Remove the file extension from the image name
            img_list_names_without_ext = [os.path.splitext(img_name)[0] for img_name in img_list_copy]

            if entry.name not in img_list_names_without_ext:
                shutil.rmtree(entry.path, ignore_errors=True)
                
    print(f'Number of already reprojected panos: {len(os.listdir(directory))}')

    # Check if we already projected images in the output directory. If so, remove them from the list
    # Check also if there are 4 files inside the directory, if not, don't remove it from the list
    img_list_copy = img_list.copy()
    for img in img_list_copy:
        img_name = img.split('.')[0]
        if os.path.exists(os.path.join(directory, img_name)):
            if len(os.listdir(os.path.join(directory, img_name))) == 4:
                img_list.remove(img)

    print('Number of panos after filtering: {}'.format(len(img_list)))

    print(f'Reprojecting panos in {args.input_dir}...')
    # Define the function to be executed in parallel
    def reproject_image(img):
        split(args, img, directory)

    # Calculate max_threads based on CPU capacity
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent()
    max_threads = int((cpu_count * (1 - cpu_percent / 100)) * 0.5)
    print(f'Using {max_threads} threads')

    # Use ThreadPoolExecutor to limit the number of concurrent threads
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        list(tqdm(executor.map(reproject_image, img_list), total=len(img_list)))

    print(f'Done reprojecting {len(img_list)} images!')

    print(f'Saving .csv file with reprojected panos names...')

    csv_file = 'reprojected_panos.csv'

    panos = []

    # Iterate through the folders in the directory and collect their names
    for entry in os.scandir(directory):
        if entry.is_dir():
            panos.append(entry.name)

    # Check if the .csv file exists
    if os.path.exists(csv_file):
        # Read the existing .csv file
        df = pd.read_csv(csv_file)
        
        # Append the new folder names only if they are not already present
        for pano in panos:
            if pano not in df['pano_id'].values:
                df = df.append({'pano_id': pano}, ignore_index=True)
    else:
        # Create a new DataFrame with the folder names and the header
        df = pd.DataFrame(panos, columns=['pano_id'])

    # Save the DataFrame to the .csv file
    df_path = os.path.join(root_dir, csv_file)
    df.to_csv(df_path, index=False)

    print(f"Reprojected panos saved to '{df_path}'.")

    return


def main(args):

    root_dir = args.input_dir

    # Replace everything that is not a character with an underscore in neighbourhood string, and make it lowercase
    args.neighbourhood = re.sub(r'[^a-zA-Z]', '_', args.neighbourhood).lower()

    if args.seg:
        reoriented_folder = 'reoriented_seg'
    else:
        reoriented_folder = 'reoriented'

    if args.ps:
        args.input_dir = os.path.join(args.input_dir, reoriented_folder)
    else:
        args.input_dir = os.path.join(args.input_dir, args.neighbourhood)
        args.input_dir = args.input_dir + '_' + args.quality
        args.input_dir = os.path.join(args.input_dir, reoriented_folder)

    reproject_panos(args, root_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--input_dir', type=str, default = 'res/dataset')
    parser.add_argument('--neighbourhood', type=str, default='osdorp')
    parser.add_argument('--quality', type=str, default='full')
    parser.add_argument('--size', type=int, default = 512)
    parser.add_argument('--ps', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--seg', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--output_dir', type=str, default = 'reprojected_seg')
    
    args = parser.parse_args()
    
    main(args)