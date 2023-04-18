'''Script that scrapes the panorama headings and lat/long coordinates of the panoramas downloaded
from ProjectSidewalk API using DownloadRunner.py in lib/sidewalk-panorama-tools.
API endpoint: https://sidewalk-amsterdam.cs.washington.edu/adminapi/panos

@author: Andrea Lombardo
'''

import subprocess
import argparse
import os
import http.client
import json
from tqdm import tqdm
from PIL import Image
import concurrent.futures
import psutil
import pandas as pd

def move_panos_to_root(input_dir):
    '''Move the panos from the subfolders to the root of the folder.'''
    # Loop through each folder in the source folder
    counter = 0
    print(f'Moving panos to root of {input_dir}...')
    for foldername in tqdm(os.listdir(input_dir)):
        folderpath = os.path.join(input_dir, foldername)

        # Skip any non-folder items in the source folder and any of the folders reoriented, reprojected, masks, backprojected
        if not os.path.isdir(folderpath) or foldername in ['reoriented','reprojected','masks','backprojected','evaluation', \
                                                           'ground_seg_outputs', 'reoriented_seg', 'reprojected_seg', 'masks_seg', \
                                                            'backprojected_seg', 'evaluation_seg']:
            continue

        # Loop through each file in the folder
        for filename in os.listdir(folderpath):

            # Skip any non-image files
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                continue

            # Get the path of the pano
            pano_path = os.path.join(folderpath, filename)

            # Move the pano to the root of the folder
            os.rename(pano_path, os.path.join(input_dir, filename))
            counter += 1

        # Delete the folder where the pano was located
        os.rmdir(folderpath)
    print(f'Moved {counter} panos to root of {input_dir}.')
    print('Done!')

'''Credits to Project Sidewealk for the following function:
https://github.com/ProjectSidewalk/sidewalk-panorama-tools/blob/master/DownloadRunner.py#L166'''
def fetch_pano_ids_from_webserver():
    unique_ids = set()
    pano_info = []
    sidewalk_server_fqdn = "sidewalk-amsterdam.cs.washington.edu"
    conn = http.client.HTTPSConnection(sidewalk_server_fqdn)
    conn.request("GET", "/adminapi/panos")
    r1 = conn.getresponse()
    data = r1.read()
    jsondata = json.loads(data)

    # Structure of JSON data
    # [
    #     {
    #         "gsv_panorama_id": "example-id",
    #         "image_width": 16384,
    #         "image_height": 8192,
    #         "panorama_lat": 52.315662,
    #         "panorama_lng":4.9531154,
    #         "photographer_heading":1.0355758,
    #         "photographer_pitch":240.7730102
    #     },
    #     ...
    # ]
    for value in jsondata:
        pano_id = value["gsv_panorama_id"]
        if pano_id not in unique_ids:
            # Check if the pano_id is an empty string.
            if pano_id and pano_id != 'tutorial':
                unique_ids.add(pano_id)
                pano_info.append(value)
            else:
                print("Pano ID is an empty string or is for tutorial")
        else:
            print("Duplicate pano ID")
    assert len(unique_ids) == len(pano_info)
    #print("Number of unique pano IDs: {}".format(len(unique_ids)))
    #print(pano_info[:10])
    return pano_info

def resize_panos(args, size):
    print(f'Resizing panos in {args.input_dir} to {size}...')

    def resize_image(filename):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            return

        pano_path = os.path.join(args.input_dir, filename)

        img = Image.open(pano_path)
        if img.size == size:
            return
        resized_img = img.resize(size, Image.Resampling.LANCZOS)
        resized_img.save(pano_path)

    # Calculate max_threads based on CPU capacity
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent()
    max_threads = int((cpu_count * (1 - cpu_percent / 100)) * 0.9)

    # Use ThreadPoolExecutor to limit the number of concurrent threads
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        list(tqdm(executor.map(resize_image, os.listdir(args.input_dir)), total=len(os.listdir(args.input_dir))))

    print('Resizing done!')

def scrape_metadata(args):
    pano_ids = []
    # Loop through each pano in the input directory
    for filename in os.listdir(args.input_dir):

        # Skip any non-image files
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            continue

        # Get the ID of the pano
        pano_id = filename.split('.')[0]
        pano_ids.append(pano_id)

    # Print the first ID
    print(f'First ID: {pano_ids[:1]}')
    print(f'# of IDs: {len(pano_ids)}')

    # Connect to the API endpoint, collect information for each pano, get the heading and save the couple ID-heading in a .csv file
    pano_info = fetch_pano_ids_from_webserver()

    # Print length of pano_info
    print(f'# of IDs in pano_info before filtering: {len(pano_info)}')

    # Filter pano_info to only include the IDs we have in our dataset
    pano_info = [pano for pano in pano_info if pano['gsv_panorama_id'] in pano_ids]

    # Print length of pano_info
    print(f'# of IDs in pano_info after filtering: {len(pano_info)}')
    assert(len(pano_info) == len(pano_ids))

    # Make a .csv file with the IDs and their headings
    csv_file_path = f'{args.input_dir}/panos.csv'

    # Write the headers to the CSV file if the file doesn't exist yet
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w') as csv_file:
            csv_file.write('pano_id,heading\n')

    # Loop through each instance of data and append it to the CSV file
    for pano in tqdm(pano_info):
        # Check if pano['gsv_panorama_id'] is in the CSV file already. If it is, skip it.
        with open(csv_file_path, mode='r') as csv_file:
            if pano['gsv_panorama_id'] in csv_file.read():
                continue
        try:
            with open(csv_file_path, mode='a') as csv_file:
                csv_file.write(f"{pano['gsv_panorama_id']},{pano['camera_heading']}\n")
        except:
            print(f"Error writing {pano['gsv_panorama_id']} to the CSV file.")

    print(f'Done! panos.csv saved in {args.input_dir}.')

    # Make another .csv file with the IDs and their latitudes and longitudes
    csv_coords_file_path = f'{args.input_dir}/panos_coords.csv'

    # Write the headers to the CSV file if the file doesn't exist yet
    if not os.path.exists(csv_coords_file_path):
        with open(csv_coords_file_path, mode='w') as csv_file:
            csv_file.write('pano_id,lat,lng\n')

    # Loop through each instance of data and append it to the CSV file
    for pano in tqdm(pano_info):
        # Check if pano['gsv_panorama_id'] is in the CSV file already. If it is, skip it.
        with open(csv_file_path, mode='r') as csv_file:
            if pano['gsv_panorama_id'] in csv_file.read():
                continue
        try:
            with open(csv_coords_file_path, mode='a') as csv_file:
                csv_file.write(f"{pano['gsv_panorama_id']},{pano['lat']},{pano['lng']}\n")
        except:
            print(f"Error writing {pano['gsv_panorama_id']} to the CSV file.")

    print(f'Done! panos_coords.csv saved in {args.input_dir}.')

def save_reoriented_panos(args):
    # Change the folder path to your desired folder
    image_folder_path = os.path.join(args.input_dir, 'reoriented')

    # Collect the names of the images without the extension
    image_names = []
    for file in os.listdir(image_folder_path):
        file_name, file_extension = os.path.splitext(file)
        if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            image_names.append(file_name)

    # Read the .csv file
    panos_csv = f'{args.input_dir}/panos.csv'
    df_panos = pd.read_csv(panos_csv)

    # Filter the rows based on the image names collection
    filtered_df_panos = df_panos[df_panos['pano_id'].isin(image_names)]
    # Remove duplicates based on the pano_id column
    filtered_df_panos = filtered_df_panos.drop_duplicates(subset=['pano_id'])

    # Save the filtered dataframe to a new .csv file in image_folder_path
    filtered_df_panos.to_csv(f'{image_folder_path}/reoriented_panos.csv', index=False)

    print(f"reoriented_panos.csv saved in {image_folder_path}")


def main(args):
    # DownloadRunner.py downloads each pano inside a folder. We need to move
    # the images to the root of the folder.
    move_panos_to_root(args.input_dir)

    # Resize panos from 16384x8192 to 2000x1000
    resize_panos(args, (2000, 1000))

    # Scrape the metadata of the panos (heading, lat/lng)
    scrape_metadata(args)

    # Reorient the panos
    subprocess.run(["python", "reorient_panos.py", f'--input_dir={args.input_dir}'])

    # Create a .csv file with the reoriented panos
    save_reoriented_panos(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='res/dataset', help='input directory')

    args = parser.parse_args()

    main(args)