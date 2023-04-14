'''Script that scrapes the panorama headings and lat/long coordinates of the panoramas downloaded
from ProjectSidewalk API using DownloadRunner.py in lib/sidewalk-panorama-tools.
API endpoint: https://sidewalk-amsterdam.cs.washington.edu/adminapi/panos

@author: Andrea Lombardo
'''

import argparse
import os
import http.client
import json
from tqdm import tqdm
from pprint import pprint

def move_panos_to_root(input_dir):
    '''Move the panos from the subfolders to the root of the folder.'''
    # Loop through each folder in the source folder
    counter = 0
    print(f'Moving panos to root of {input_dir}...')
    for foldername in tqdm(os.listdir(input_dir)):
        folderpath = os.path.join(input_dir, foldername)

        # Skip any non-folder items in the source folder
        if not os.path.isdir(folderpath):
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

def main(args):

    # DownloadRunner.py downloads each pano inside a folder. We need to move
    # the images to the root of the folder.
    move_panos_to_root(args.input_dir)

    # Loop through args.input_dir and collect the IDs of the panos
    pano_ids = []
    # Loop through each folder in the source folder
    for filename in os.listdir(args.input_dir):

        # Skip any non-image files
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            continue

        # Get the ID of the pano
        pano_id = filename.split('.')[0]
        pano_ids.append(pano_id)

    # Print the first ID
    print(f'First ID: {pano_ids[:1]}')
    pprint(pano_ids[:10])
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

    '''# Make a .csv file with the IDs and their headings
    with open(f'{args.input_dir}/panos.csv', mode='w') as csv_file:
        # Write the headers to the CSV file
        csv_file.write('pano_id,heading\n')

        # Loop through each instance of data and write it to the CSV file
        for pano in tqdm(pano_info):
            try:
                csv_file.write(f"{pano['gsv_panorama_id']},{pano['photographer_heading']}\n")
            except:
                print(f"Error writing {pano['gsv_panorama_id']} to the CSV file.")

        print(f'Done! panos.csv saved in {args.input_dir}.')

    # Make another .csv file with the IDs and their latitudes and longitudes
    with open(f'{args.input_dir}/panos_coords.csv', mode='w') as csv_file:
        # Write the headers to the CSV file
        csv_file.write('pano_id,lat,lng\n')

        # Loop through each instance of data and write it to the CSV file
        for pano in tqdm(pano_info):
            try:
                csv_file.write(f"{pano['gsv_panorama_id']},{pano['panorama_lat']},{pano['panorama_lng']}\n")
            except:
                print(f"Error writing {pano['gsv_panorama_id']} to the CSV file.")

        print(f'Done! panos_coords.csv saved in {args.input_dir}.')'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='res/dataset', help='input directory')

    args = parser.parse_args()

    main(args)