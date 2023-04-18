import cv2
import json
import numpy as np
import argparse
import re
import os

def blacken_labels(input_image_path, masks_path, json_data, labels_to_blacken, output_image_path):
    # Load the panoramic image and create a copy
    image = cv2.imread(input_image_path)
    image_copy = np.copy(image)

    # Load the masks as an image
    masks = cv2.imread(masks_path)

    # Iterate through the JSON data and find the labels to blacken
    for item in json_data:
        if item['label'] in labels_to_blacken:
            # Create a binary mask using the color information
            color = np.array(item['color']) * 255
            binary_mask = np.all(masks == color, axis=-1)

            # Multiply the binary mask with the copied image
            image_copy[binary_mask] = 0

    # Save the modified image
    cv2.imwrite(output_image_path, image_copy)    


def main(args):
    # Replace everything that is not a character with an underscore in neighbourhood string, and make it lowercase
    args.neighbourhood = re.sub(r'[^a-zA-Z]', '_', args.neighbourhood).lower()
    
    # Go through each folder in the input directory
    for folder in os.listdir(args.input_dir):
        base_path = os.path.join(args.input_dir, folder)
        blacken_labels(input_image_path, masks_path, json_data, labels_to_blacken, output_image_path)
    
    
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='res/dataset')
    parser.add_argument('--neighbourhood', type=str, default='osdorp')
    parser.add_argument('--quality', type=str, default='full')
    parser.add_argument('--visualize', type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--threshold', type=int, default=100, help='Threshold for distance-based metrics (in pixels)')
    parser.add_argument('--radius', type=int, default=100, help='Radius for point-to-mask best IoU (in pixels)')
    parser.add_argument('--ps', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--seg', type=bool, default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    main(args)

    json_str = '...'
    json_data = json.loads(json_str)
    labels_to_blacken = ['sky', 'car']
    input_image_path = 'path/to/panoramic_image.jpg'
    masks_path = 'path/to/masks_image.jpg'
    output_image_path = 'path/to/output_image.jpg'

    blacken_labels(input_image_path, masks_path, json_data, labels_to_blacken, output_image_path)