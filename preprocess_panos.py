'''Script that extracts the panoramas from ProjectSidewalk download folders, put them in a single folder,
creates a list of their IDs, scrape the panorama headings and lat/long coordinates from ProjectSidewalk API.
API endpoint: https://sidewalk-amsterdam.cs.washington.edu/adminapi/panos

@author: Andrea Lombardo
'''

import argparse
import os
import re
import shutil
import http.client
import json
import pandas as pd

import psutil
import concurrent.futures
from tqdm import tqdm

def move_panos(source_folder, destination_folder):
    # Make a counter for each image the script finds
    counter = 0

    # Loop through each folder in the source folder
    for foldername in tqdm(os.listdir(source_folder)):
        folderpath = os.path.join(source_folder, foldername)

        # Skip any non-folder items in the source folder
        if not os.path.isdir(folderpath):
            continue

        # Loop through each file in the folder
        for filename in os.listdir(folderpath):
            filepath = os.path.join(folderpath, filename)

            # Skip any non-image files
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                continue

            # Move the image to the destination folder
            shutil.move(filepath, destination_folder)
            counter += 1
    print("Moved {} images to {}.".format(counter, destination_folder))
    print("Done!")

# Credits to Project Sidewalk for the following function
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

def main(args):

    # Set the source and destination folder paths
    source_folder = "download_data"
    
    # Make a directory specific for the neighbourhood
    # Save the panoramas in the directory

    # Replace everything that is not a character with an underscore in neighbourhood string, and make it lowercase
    args.neighbourhood = re.sub(r'[^a-zA-Z]', '_', args.neighbourhood).lower()

    destination_folder = os.path.join(args.output_dir, args.neighbourhood)
    destination_folder = destination_folder + '_' + args.quality
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)

    move_panos(source_folder, destination_folder)

    # Loop through destination_folder and collect the IDs of the panos
    pano_ids = []
    # Loop through each folder in the source folder
    for filename in os.listdir(destination_folder):

        # Skip any non-image files
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            continue

        # Get the ID of the pano
        pano_id = filename.split('.')[0]
        pano_ids.append(pano_id)

    # Print the first ID
    print(f'First ID: {pano_ids[:1]}')

    # Connect to the API endpoint, collect information for each pano, get the heading and save the couple ID-heading in a .csv file
    pano_info = fetch_pano_ids_from_webserver()

    # Filter pano_info to only include the IDs we have in our dataset
    pano_info = [pano for pano in pano_info if pano['gsv_panorama_id'] in pano_ids]

    # Make a .csv file with the IDs and their headings
    with open(f'{destination_folder}/panos.csv', mode='w') as csv_file:
        # Write the headers to the CSV file
        csv_file.write('pano_id,heading\n')

        # Loop through each instance of data and write it to the CSV file
        for pano in tqdm(pano_info):
            try:
                csv_file.write(f"{pano['gsv_panorama_id']},{pano['photographer_heading']}\n")
            except:
                print(f"Error writing {pano['gsv_panorama_id']} to the CSV file.")

    # Make another .csv file with the IDs and their latitudes and longitudes
    with open(f'{destination_folder}/panos_coords.csv', mode='w') as csv_file:
        # Write the headers to the CSV file
        csv_file.write('pano_id,lat,lng\n')

        # Loop through each instance of data and write it to the CSV file
        for pano in tqdm(pano_info):
            try:
                csv_file.write(f"{pano['gsv_panorama_id']},{pano['panorama_lat']},{pano['panorama_lng']}\n")
            except:
                print(f"Error writing {pano['gsv_panorama_id']} to the CSV file.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='res/dataset', help='input directory')
    parser.add_argument('--neighbourhood', type=str, default='Osdorp', help='neighbourhood')
    parser.add_argument('--quality', type=str, default='small', help='quality of the images')

    args = parser.parse_args()

    main(args)