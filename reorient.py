''' 
Project panoramic images to cubemaps and save front, back, left, right as png files.
Input: a folder with panoramic (Equirectangular) images
Output: a folder with cubemaps

@author: Andrea Lombardo
'''

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

from tqdm import tqdm

def orient_panorama(img, heading):
    '''Credits to Tim Alpherts for the function'''
    # Reshift panorama according to heading
    shift = heading/360
    pixel_split = int(img.size[0] * shift)
    
    left = Image.fromarray(np.array(img)[:, pixel_split:])
    right = Image.fromarray(np.array(img)[:, :pixel_split])
    
    reoriented_img = np.hstack((left, right))
    
    return Image.fromarray(reoriented_img)

def orient_panoramas(input_dir, output_dir):
    # Define path to .csv 
    csvpath = input_dir + 'panos.csv'
    csv = pd.read_csv(csvpath)

    # Make a folder to store the reoriented panoramas called sample_dataset_reoriented
    folder_reoriented = output_dir
    os.makedirs(folder_reoriented, exist_ok=True)

    # Reorient all the images in the .csv file and save them in a new folder
    # Use tqdm to track progress
    for index, row in tqdm(csv.iterrows()):
        img_filename = row['pano_id']
        
        img = Image.open(input_dir + img_filename + '.jpg', formats=['JPEG'])
        heading = row['heading']
        reoriented_img = orient_panorama(img, heading)
        reoriented_img.save(folder_reoriented + '/' + img_filename, format='JPEG')
    return

def main(args):
    orient_panoramas(args.input_dir, args.output_dir)
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--input_dir', type=str, default = 'res/dataset/') 
    parser.add_argument('--output_dir', type=str, default = 'res/dataset/reoriented')
    
    args = parser.parse_args()
    
    main(args)