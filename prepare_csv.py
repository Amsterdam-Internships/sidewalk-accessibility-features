'''Script to create a list of .csv files used during the pipeline.
1. panos_hq.csv: list of panos with obstacle labels associated. The list will be used to filter the panos during the API download
process (in download_panos.py).
2. labels.csv: list of obstacle labels associated to each pano. The list will be used throughout the pipeline for various steps.'''

import requests
import geopandas as gpd
import json
import pandas as pd
import os
import argparse

def get_ps_labels():

    base_url = "https://sidewalk-amsterdam.cs.washington.edu/v2/access/attributesWithLabels?lat1={}&lng1={}&lat2={}&lng2={}" 
    
    whole = (52.303, 4.8, 52.425, 5.05)
    centrum_west = (52.364925, 4.87444, 52.388692, 4.90641)
    test = (52.0, 4.0, 53.0, 5.0)

    coords = test

    url = base_url.format(*coords)

    project_sidewalk_labels = requests.get(url.format(*coords)).json()

    ps_labels_df = gpd.GeoDataFrame.from_features(project_sidewalk_labels['features'])
    # Print length before filtering
    print('Length raw labels: ', len(ps_labels_df))

    return ps_labels_df

def get_xy_coords_ps_labels():
    # Send a get call to this API: https://sidewalk-amsterdam-test.cs.washington.edu/adminapi/labels/cvMetadata
    other_labels = requests.get('https://sidewalk-amsterdam.cs.washington.edu/adminapi/labels/cvMetadata').json()
    other_labels_df = pd.DataFrame(other_labels)
    print('Length raw labels (2): ', len(other_labels_df))

    return other_labels_df

def main(args):
    # Get Project Sidewalk labels from API
    ps_labels_df = get_ps_labels()
    # Get the labels from another API call
    other_labels_df = get_xy_coords_ps_labels()

    # Filter other_labels_df to only contain obstacles ('label_type_id' == 3)
    other_labels_df = other_labels_df[other_labels_df['label_type_id'] == 3]

    print(f'Number of obstacles labels from cvMetadata API (low quality): {len(other_labels_df)}')

    # Intersect other_labels_df with ps_labels_df
    other_labels_df = other_labels_df[other_labels_df['label_id'].isin(ps_labels_df['label_id'])]
    print(f'Number of obstacles labels in other_labels_df after filtering for high quality data: {len(other_labels_df)}')

    # Count the number of panos (unique gsv_panorama_id)
    print(f'Number of panos with obstacles labels: {len(other_labels_df["gsv_panorama_id"].unique())}')

    # 2. labels.csv
    # Save the filtered labels to a .csv file called labels.csv
    labels_path = os.path.join(args.output_dir, 'labels.csv')
    other_labels_df.to_csv(labels_path, index=False)

    # 1. panos_hq.csv
    # Save the filtered panos to a .csv file called panos_hq.csv using .unique
    # Replace "'" with "" in the gsv_panorama_id column
    other_labels_df['gsv_panorama_id'] = other_labels_df['gsv_panorama_id'].str.replace("'", "")
    # Save the filtered panos to a .csv file called panos_hq.csv, with each line containing a unique
    # gsv_panorama_id. First, convert it to a panda dataframe. Then, add a header called 'pano_id'.
    panos_path = os.path.join(args.output_dir, 'panos_hq.csv')
    pd.DataFrame(other_labels_df['gsv_panorama_id'].unique(), columns=['pano_id']).to_csv(panos_path, index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--output_dir', type=str, default = 'download_PS')
    
    args = parser.parse_args()
    
    main(args)

