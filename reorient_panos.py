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
import concurrent.futures
import psutil

from PIL import Image

from tqdm import tqdm

def orient_single_pano(img, heading):
    '''Credits to Tim Alpherts for the function'''

    # We select a tolerance of 0.1 degrees so that 
    # the panorama is not reoriented if the heading is close to 0
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
            print('Tile cannot extend outside image')
            print('Panorama not reoriented: ' + img.filename)
            return img, False


def orient_panos(args):

     # Define path to .csv
    if args.ps:
        path = args.input_dir
    else:
        path = os.path.join(args.input_dir, args.neighbourhood)
        path = path + '_' + args.quality
    
    csvpath = os.path.join(path, 'panos.csv')
    csv = pd.read_csv(csvpath)

    print(f'Number of panos in panos.csv: {len(csv)}')

    # Define output directory
    directory = os.path.join(path, 'reoriented')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Collect the list of images already reoriented in directory. Remove them from the pandas dataframe
    reoriented_list = os.listdir(directory)

    # Remove the extension from the list using split
    reoriented_list = [reoriented_list[i].split('.')[0] for i in range(len(reoriented_list))]

    csv = csv[~csv['pano_id'].isin(reoriented_list)]

    print(f'Number of panos left to reorient: {len(csv)}')

    def process_image(index, row):
        img_filename = row['pano_id']
        
        img = Image.open(os.path.join(path, img_filename) + '.jpg', formats=['JPEG'])

        # Temporary check to see if the image has not been resized
        if img.size != (2000,1000):
            print('Image not resized: ' + img_filename)
            return
        else:
            heading = row['heading']
            reoriented_img, bool = orient_single_pano(img, heading)
        if not (bool):
            # Save in a .csv file the panoramas that were not reoriented
            # Make sure to go a new line after each entry
            with open(os.path.join(path, 'not_reoriented.csv'), 'a') as f:
                f.write(img_filename + ',' + str(heading) + '\n')

        # Save the image in the reoriented folder
        reoriented_img.save(os.path.join(directory, img_filename) + '.jpg', 'JPEG')
        # Cancel the original image with name "img_filename" and path "path". Check first if it exists
        if os.path.exists(os.path.join(path, img_filename) + '.jpg'):
            os.remove(os.path.join(path, img_filename) + '.jpg')

    # Calculate max_threads based on CPU capacity
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent()
    max_threads = int((cpu_count * (1 - cpu_percent / 100)) * 0.9)

    # Use ThreadPoolExecutor to limit the number of concurrent threads
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        list(tqdm(executor.map(lambda x: process_image(*x), csv.iterrows()), total=len(csv)))

    return

def main(args):
    # Replace everything that is not a character with an underscore in neighbourhood string, and make it lowercase
    args.neighbourhood = re.sub(r'[^a-zA-Z]', '_', args.neighbourhood).lower()
    
    orient_panos(args)
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--input_dir', type=str, default = 'res/dataset') 
    parser.add_argument('--quality', type=str, default='full')
    parser.add_argument('--neighbourhood', type=str, default='osdorp')
    parser.add_argument('--ps', type=bool, default=True, action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    
    main(args)