'''Script that extracts the left and right images from each reprojected image folder
and saves them in a new folder called their folder name + "left_right" in res/dataset/segmented.

@author: Andrea Lombardo
'''

import argparse
import os
import re

import psutil
import concurrent.futures
from tqdm import tqdm

import mmcv
import mmseg
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

def extract_left_right(args, img_path):
    input_dir = os.path.join(args.input_dir, img_path)
    _, _, left, right = os.listdir(input_dir)
    left, right = os.path.join(input_dir, left), os.path.join(input_dir, right)
    return left, right

def segment(args, img_path, output_directory, model):

    # Extract left and right images
    left, right = extract_left_right(args, img_path)
    left_img = mmcv.imread(left)
    right_img = mmcv.imread(right)

    # Segment left and right images
    left_result = inference_segmentor(model, left_img)
    right_result = inference_segmentor(model, right_img)
    # Save segmented images
    model.show_result(left, left_result, show=False, out_file=os.path.join(output_directory, img_path, 'left.png'))
    model.show_result(right, right_result, show=False, out_file=os.path.join(output_directory, img_path, 'right.png'))
    
    # Extract the sidewalks and save them in the same folder
    # Turn every pixel of img to black if is not in the sidewalk class (1)
    left_extracted = left_img.copy()
    left_extracted[left_result[0] != 1] = 0

    right_extracted = right_img.copy()
    right_extracted[right_result[0] != 1] = 0
    # Save the images
    mmcv.imwrite(left_extracted, os.path.join(output_directory, img_path, 'left_sidewalk.png'))
    mmcv.imwrite(right_extracted, os.path.join(output_directory, img_path, 'right_sidewalk.png'))

def main(args):

    # Replace everything that is not a character with an underscore in neighbourhood string, and make it lowercase
    args.neighbourhood = re.sub(r'[^a-zA-Z]', '_', args.neighbourhood).lower()

    # Add the neighbourhood name to the path
    args.input_dir = os.path.join(args.input_dir, args.neighbourhood)
    # Add 'reprojected' to the path
    args.input_dir = os.path.join(args.input_dir, 'reprojected')
    print(args.input_dir)

    # Create the output directory if it doesn't exist
    # Take args.input_dir, strip the last part of the path and add reprojected
    output_directory = os.path.join(os.path.dirname(args.input_dir), 'segmented')
    print(output_directory)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Define list of images in the input directory
    img_list = os.listdir(args.input_dir)
    print(img_list)

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config_file, args.checkpoint_file, device='cuda:0')

    # Create a for loop where each iteration calls segment(args, img) for each image in img_list2
    # Use tqdm to show a progress bar
    for img in tqdm(img_list):
        segment(args, img, output_directory, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='res/dataset', help='input directory')
    parser.add_argument('--neighbourhood', type=str, default='Osdorp', help='neighbourhood')
    parser.add_argument('--config_file', type=str, default='lib/mmsegmentation/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py', help='config file')
    parser.add_argument('--checkpoint_file', type=str, default='lib/mmsegmentation/checkpoints/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth', help='checkpoint file')

    args = parser.parse_args()

    main(args)