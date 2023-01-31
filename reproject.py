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

def split(args, img_path):

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
    directory = os.path.join(os.path.dirname(args.input_dir), 'reprojected')
    directory = os.path.join(directory, img_path)
    if not os.path.exists(directory):
            os.makedirs(directory)
    # save them in that dir
    front.save(os.path.join(directory, 'front.png'))
    right.save(os.path.join(directory, 'right.png'))
    back.save(os.path.join(directory, 'back.png'))
    left.save(os.path.join(directory, 'left.png'))

    print('saved {}!'.format(img_path))

def main(args):

    # Replace everything that is not a character with an underscore in neighbourhood string, and make it lowercase
    args.neighbourhood = re.sub(r'[^a-zA-Z]', '_', args.neighbourhood).lower()
    args.input_dir = os.path.join(args.input_dir, args.neighbourhood)
    args.input_dir = os.path.join(args.input_dir, 'reoriented')

    cpu_percent_threshold = 90
    mem_percent_threshold = 90
    disk_percent_threshold = 90
    # Using all the available cores for the thread pool makes the system crush,
    # su we use only half of them
    num_workers = int(psutil.cpu_count()/2)
    print('Number of workers: {}'.format(num_workers))

    # Define list of images in the input directory
    img_list = os.listdir(args.input_dir)

    # Limit the number of images for testing
    # Randomly choose 100 in the list and make it a list, using the same seed for reproduction
    np.random.seed(42)
    img_list2 = np.random.choice(img_list, 100, replace=False).tolist()

    # Create a list of tuples called inputs where each of them is composed of args and one image in img_list2
    inputs = [(args, img) for img in img_list2]
    
    # Create the output directory if it doesn't exist
    # Take args.input_dir, strip the last part of the path and add reprojected
    directory = os.path.join(os.path.dirname(args.input_dir), 'reprojected')

    if not os.path.exists(directory):
                os.makedirs(directory)

    # Create a thread pool executor with 4 worker threads
    # we create a list of Future object by calling executor.submit(split, x, y) for each tuple (x, y) in inputs.
    # Then we use the concurrent.futures.as_completed(futures) function to iterate over the Future objects
    # as they complete and retrieve the result.
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(split, x, y) for x, y in inputs]
        # Use cpu_percent to monitor the CPU usage and dynamically adjust the number of workers
        # Do the same for memory usage and disk usage
        while True:
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > cpu_percent_threshold:
                num_workers -= 1
                print('New number of workers: {}'.format(num_workers))
                executor._max_workers = max(num_workers, 0)
            mem_percent = psutil.virtual_memory().percent
            if mem_percent > mem_percent_threshold:
                num_workers -= 1
                executor._max_workers = max(num_workers, 0)
            disk_percent = psutil.disk_usage("/").percent
            if disk_percent > disk_percent_threshold:
                num_workers -= 1
                executor._max_workers = max(num_workers, 0)
            if all(future.done() for future in futures):
                break
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    result = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (input, exc))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--input_dir', type=str, default = 'res/dataset')
    parser.add_argument('--neighbourhood', type=str, default='osdorp')
    parser.add_argument('--size', type=int, default = 512)
    
    args = parser.parse_args()
    
    main(args)