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

import numpy as np
from PIL import Image
from tqdm import tqdm     

def split(args, img_path, directory):

    print(f'Processing image...')
    img = os.path.join(args.input_dir, img_path)

    # VrProjector the images 
    size = args.size
    eq = vrProjector.EquirectangularProjection()
    eq.loadImage(img)
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

def reproject_panos(args):
    # Define list of images in the input directory
    img_list = os.listdir(args.input_dir)
    print('Number of reoriented panos: {}'.format(len(img_list)))

    # Define output directory
    directory = os.path.join(os.path.dirname(args.input_dir), 'reprojected')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Check if we already projected images in the output directory. If so, remove them from the list
    # Check also if there are 4 files inside the directory, if not, don't remove it from the list
    for img in img_list:
        if os.path.exists(os.path.join(directory, img)):
            if len(os.listdir(os.path.join(directory, img))) == 4:
                img_list.remove(img)

    print('Number of panos after filtering: {}'.format(len(img_list)))

    print(f'Reprojecting panos in {args.input_dir}...')
    # Define the function to be executed in parallel
    def reproject_image(img):
        split(args, img, directory)

    # Calculate max_threads based on CPU capacity
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent()
    max_threads = int((cpu_count * (1 - cpu_percent / 100)) * 0.2)
    print(f'Using {max_threads} threads')

    # Use ThreadPoolExecutor to limit the number of concurrent threads
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        list(tqdm(executor.map(reproject_image, img_list), total=len(img_list)))
    
    print(f'Done reprojecting {len(img_list)} images!')
    return


def main(args):

    # Replace everything that is not a character with an underscore in neighbourhood string, and make it lowercase
    args.neighbourhood = re.sub(r'[^a-zA-Z]', '_', args.neighbourhood).lower()

    if args.ps:
        args.input_dir = os.path.join(args.input_dir, 'reoriented')
    else:
        args.input_dir = os.path.join(args.input_dir, args.neighbourhood)
        args.input_dir = args.input_dir + '_' + args.quality
        args.input_dir = os.path.join(args.input_dir, 'reoriented')

    reproject_panos(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--input_dir', type=str, default = 'res/dataset')
    parser.add_argument('--neighbourhood', type=str, default='osdorp')
    parser.add_argument('--quality', type=str, default='full')
    parser.add_argument('--size', type=int, default = 512)
    parser.add_argument('--ps', type=bool, default=True, action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    
    main(args)