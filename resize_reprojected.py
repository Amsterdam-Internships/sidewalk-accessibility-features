'''Script that traverse a list of directories in --input.dir that contain images (.png) and resizes them to --size.
It outputs the resized images to --output_dir following the same folder structure of --input.dir.'''

import argparse
import os
from tqdm import tqdm

import cv2

def resize_panos(args):
    print(f'Resizing panos to {args.size} x {args.size} pixels...')
    # args.input_dir contains the pano folders with the masks
    # args.output_dir will contain the resized masks
    directory = args.output_dir

    # For each pano folder in args.input_dir, go through the masks, resize them and save them in output_dir/pano_folder
    for pano in tqdm(os.listdir(args.input_dir)):
        panos_path = os.path.join(args.input_dir, pano)
        faces = os.listdir(panos_path)

        # Create the pano folder in the output directory if it doesn't exist
        if not os.path.exists(os.path.join(directory, pano)):
            os.makedirs(os.path.join(directory, pano))

        for face_file in faces:
            # Load the face
            face = cv2.imread(os.path.join(panos_path, face_file))
            # Resize the face
            face = cv2.resize(face, (args.size, args.size))
            # Save the face
            cv2.imwrite(os.path.join(directory, pano, face_file), face)
    print('Done!')

def main(args):
    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Input directory: ', args.input_dir)
    print('Output directory: ', args.output_dir)

    # 1. Resize masks to args.size x args.size
    resize_panos(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='res/dataset', help='input directory')
    parser.add_argument('--size', type=int, default=2048, help='resize according to face size')
    parser.add_argument('--output_dir', type=str, default='res/dataset', help='output directory')

    args = parser.parse_args()

    main(args)