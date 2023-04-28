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
import requests
import geopandas as gpd
import shutil

'''TODO: Move the function in utils.py'''
def get_ps_labels(args):

    base_url = "https://sidewalk-amsterdam.cs.washington.edu/v2/access/attributesWithLabels?lat1={}&lng1={}&lat2={}&lng2={}" 
    
    whole = (52.303, 4.8, 52.425, 5.05)
    centrum_west = (52.364925, 4.87444, 52.388692, 4.90641)
    test = (52.0, 4.0, 53.0, 5.0)

    coords = test

    url = base_url.format(*coords)

    label_dump = os.path.join(args.label_dump, 'attributesWithLabels')

    try:
        project_sidewalk_labels = json.load(open(label_dump, 'r'))
    except Exception as e:
        print("Couldn't load local dump")
        project_sidewalk_labels = requests.get(url.format(*coords)).json()
        json.dump(project_sidewalk_labels, open(label_dump, 'w'))

    ps_labels_df = gpd.GeoDataFrame.from_features(project_sidewalk_labels['features'])
    # Print length before filtering
    print('Length of labels from attributesWithLabels API: ', len(ps_labels_df))

    return ps_labels_df

'''TODO: Move the function in utils.py'''
def get_xy_coords_ps_labels():
    # Send a get call to this API: https://sidewalk-amsterdam-test.cs.washington.edu/adminapi/labels/cvMetadata
    other_labels = requests.get('https://sidewalk-amsterdam.cs.washington.edu/adminapi/labels/cvMetadata').json()
    other_labels_df = pd.DataFrame(other_labels)
    print('Length raw labels from cvMetadata API (low quality) : ', len(other_labels_df))

    return other_labels_df

def move_panos_to_root(input_dir):
    '''Move the panos from the subfolders to the root of the folder.'''
    # Loop through each folder in the source folder
    counter = 0
    print(f'Moving panos to root of {input_dir}...')
    for foldername in tqdm(os.listdir(input_dir)):
        folderpath = os.path.join(input_dir, foldername)

        # Skip any non-folder items in the source folder and any folders which name contains
        # reoriented, reprojected, masks, backprojected, evaluation, ground_seg_outputs
        if not os.path.isdir(folderpath) or any(x in foldername for x in ['reoriented', \
                'reprojected', 'masks', 'backprojected', 'evaluation', 'ground_seg_outputs']):
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

def resize_panos(args):
    size = (args.pano_width, args.pano_height)
    print(f'Resizing panos in {args.input_dir} to {size}...')
    if args.blacken:
        print('Also blackening 200px from the top and 100px from the bottom of the panos.')

    def resize_image(filename):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            return

        pano_path = os.path.join(args.input_dir, filename)

        img = Image.open(pano_path)
        if img.size == size:
            return
        resized_img = img.resize(size, Image.Resampling.LANCZOS)
        if args.blacken:
            # Make 200px from the top and 100px from the bottom black
            resized_img.paste((0, 0, 0), (0, 0, size[0], 200))
            resized_img.paste((0, 0, 0), (0, size[1] - 100, size[0], size[1]))

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
    # Take the dirname of args.input_dir
    parent_dir = os.path.dirname(args.input_dir)
    reoriented_path = os.path.join(parent_dir, 'reoriented')

    # Collect the names of the images without the extension
    image_names = []
    for file in os.listdir(reoriented_path):
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

    # Save the filtered dataframe to a new .csv file in reoriented_path
    filtered_df_panos.to_csv(f'{args.input_dir}/reoriented_panos.csv', index=False)

    print(f"reoriented_panos.csv saved in {args.input_dir}")

def remove_panos_without_metadata(args):
    # Take the dirname of args.input_dir
    parent_dir = os.path.dirname(args.input_dir)
    reoriented_path = os.path.join(parent_dir, 'reoriented')

    '''Remove the images that have not been reoriented because they don't have heading information'''
    len_folder = len(os.listdir(reoriented_path))
    print(f'There are {len_folder} reoriented images in the folder')
    # Open reoriented_panos.csv and collect the list of pano_id that have been reoriented
    reoriented_csv = pd.read_csv(os.path.join(reoriented_path, 'reoriented_panos.csv'))
    reoriented_list = reoriented_csv['pano_id'].tolist()

    print(f'There are {len(reoriented_list)} reoriented images in the reoriented_panos.csv file')

    # Remove using os the images that are not in the reoriented_panos.csv
    for pano in os.listdir(reoriented_path):
        if pano.split('.')[0] not in reoriented_list:
            os.remove(reoriented_path, pano)
    new_len_folder = len(os.listdir(reoriented_path))
    print(f'After filtering, there are now {new_len_folder} reoriented images in the folder')

def save_hq_panos(args):
    '''# Get Project Sidewalk labels from API
    ps_labels_df = get_ps_labels(args)
    # Get the labels from another API endpoint
    other_labels_df = get_xy_coords_ps_labels()

    # Filter other_labels_df to only contain obstacles ('label_type_id' == 3)
    other_labels_df = other_labels_df[other_labels_df['label_type_id'] == 3]

    print(f'Number of obstacles labels from cvMetadata API (low quality): {len(other_labels_df)}')

    # Intersect other_labels_df with ps_labels_df
    other_labels_df = other_labels_df[other_labels_df['label_id'].isin(ps_labels_df['label_id'])]
    print(f'Number of obstacles labels in other_labels_df after filtering for high quality data: {len(other_labels_df)}')

    # Select the unique pano ids from gsv_panorama_id column in other_labels_df
    other_labels_df = other_labels_df['gsv_panorama_id'].unique()

    print(f'Number of panos containing only obstacles as labels: {len(other_labels_df)}')'''

    # Load labels from the CSV file
    other_labels_df = pd.read_csv(args.label_dump)

    image_folder_path = os.path.join(args.input_dir, 'reoriented')

    # Create a new folder called reoriented_hq
    if not os.path.exists(os.path.join(args.input_dir, 'reoriented_hq')):
        os.mkdir(os.path.join(args.input_dir, 'reoriented_hq'))

    # Copy the images that are in other_labels_df to a new folder called reoriented_hq
    # Make sure to filter non .jpg images
    for file in os.listdir(image_folder_path):
        file_name, file_extension = os.path.splitext(file)
        if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            if file_name in other_labels_df:
                shutil.copy(os.path.join(image_folder_path, file), os.path.join(args.input_dir, 'reoriented_hq'))

    # Check number of files in reoriented_hq folder
    print(f'Number of images in reoriented_hq folder: {len(os.listdir(os.path.join(args.input_dir, "reoriented_hq")))}')


def main(args):
    # DownloadRunner.py downloads each pano inside a folder. We need to move
    # the images to the root of the folder.
    move_panos_to_root(args.input_dir)

    # Resize panos from 16384x8192 to (args.width)x(args.height)
    resize_panos(args)

    # Scrape the metadata of the panos (heading, lat/lng)
    scrape_metadata(args)

    # Reorient the panos
    parent_dir = os.path.dirname(args.input_dir)
    subprocess.run(["python", "reorient_panos.py", f'--input_dir={parent_dir}'])

    # Create a .csv file with the reoriented panos
    save_reoriented_panos(args)

    # Remove the panos that have not been reoriented (because they don't have heading information)
    remove_panos_without_metadata(args)

    '''if args.filter:
        # Filter the panos that have high quality obstacles labels
        print('Filtering the panos that have high quality obstacles labels...')
        save_hq_panos(args)
        print('Done!')'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='res/dataset', help='input directory')
    parser.add_argument('--pano_width', type=int, default=2048, help='panorama width')
    parser.add_argument('--pano_height', type=int, default=1024, help='panorama height')
    #parser.add_argument('--filter', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--blacken', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--label_dump', type=str, default='res/labels/labels.csv')

    args = parser.parse_args()

    main(args)