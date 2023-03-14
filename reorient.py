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
import re

from PIL import Image

from tqdm import tqdm

def orient_panorama(img, heading):
    '''Credits to Tim Alpherts for the function'''

    tolerance = 0.1
    if abs(heading) <= tolerance:
        return img, False
    
    else:
        
        # Reshift panorama according to heading
        shift = heading/360
        pixel_split = int(img.size[0] * shift)
        
        try:
            left = Image.fromarray(np.array(img)[:, pixel_split:])
            right = Image.fromarray(np.array(img)[:, :pixel_split])

            reoriented_img = np.hstack((left, right))
            
            return Image.fromarray(reoriented_img), True

        except ValueError:
            # Print the error message
            print('tile cannot extend outside image')
            print('Panorama not reoriented: ' + img.filename)
            print('Shift: ' + str(shift))
            print('Pixel split: ' + str(pixel_split))
            print('Image size: ' + str(img.size))
            print('Heading: ' + str(heading))
            
            return img, False

    
    
    
    

def orient_panoramas(args):

    # Define path to .csv
    path = os.path.join(args.input_dir, args.neighbourhood)
    path = path + '_' + args.quality
    csvpath = os.path.join(path, 'panos.csv')
    csv = pd.read_csv(csvpath)

    # Make a folder to store the reoriented panoramas called sample_dataset_reoriented
    folder_reoriented = os.path.join(path, 'reoriented')
    os.makedirs(folder_reoriented, exist_ok=True)

    # Reorient all the images in the .csv file and save them in a new folder
    # Use tqdm to track progress
    for index, row in tqdm(csv.iterrows()):
        img_filename = row['pano_id']
        
        img = Image.open(os.path.join(path, img_filename) + '.jpg', formats=['JPEG'])
        heading = row['heading']
        reoriented_img, bool = orient_panorama(img, heading)
        if not (bool):
            # Save in a .csv file the panoramas that were not reoriented
            # Make sure to go a new line after each entry
            neighbourhood = args.neighbourhood + '_' + args.quality
            csv_path = os.path.join(args.input_dir, neighbourhood)
            with open(os.path.join(csv_path, 'not_reoriented.csv'), 'a') as f:
                f.write(img_filename + ',' + str(heading) + '\n')

        # Save the image in the reoriented folder
        reoriented_img.save(folder_reoriented + '/' + img_filename, format='JPEG')
    return

def main(args):
    # Replace everything that is not a character with an underscore in neighbourhood string, and make it lowercase
    args.neighbourhood = re.sub(r'[^a-zA-Z]', '_', args.neighbourhood).lower()
    
    orient_panoramas(args)
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--input_dir', type=str, default = 'res/dataset') 
    parser.add_argument('--quality', type=str, default='full')
    parser.add_argument('--neighbourhood', type=str, default='osdorp')
    
    args = parser.parse_args()
    
    main(args)