'''Evaluate the predictions by comparing them to Project Sidewalk's (PS) ground truth
labels. Run this script after backproject_masks.py.'''

import os
import pandas as pd
import geopandas as gpd
import requests
import json
import argparse
import re
import folium

def evaluate(args, directory):
    # We assume both PS labels and OD labels are in the same coordinate system
    # EPSG:4326, the WGS84 coordinate system

    predictions_path = os.path.join(args.input_dir, 'object_locations.csv')
    
    base_url = 'https://sidewalk-amsterdam.cs.washington.edu/v2/access/attributesWithLabels?' + \
'lat1={}&lng1={}&lat2={}&lng2={}'

    # Get labels from Project Sidewalk
    whole = (52.303, 4.8, 52.425, 5.05)
    #centrum_west = (52.364925, 4.87444, 52.388692, 4.90641)
    coords = whole
    url = base_url.format(*coords)
    local_dump = url.replace('/', '|')
    try:
        project_sidewalk_labels = json.load(open(local_dump, 'r'))
    except Exception as e:
        print("Couldn't load local dump")
        project_sidewalk_labels = requests.get(url.format(*coords)).json()
        json.dump(project_sidewalk_labels, open(local_dump, 'w'))

    od_labels = pd.read_csv(predictions_path)
    # Convert od_labels_path to a geopandas dataframe
    od_labels_df = gpd.GeoDataFrame(od_labels, geometry=gpd.points_from_xy(od_labels.lon, od_labels.lat), crs='epsg:4326')
    print('Number of od_labels: {}'.format(len(od_labels_df)))

    # Heuristic 1: we have access to two objects/obstacles per panorama (left/right),
    # while there is probably one PS label linked to the panorama.
    # We decide to use only one of the two objects based on pano_id
    od_labels_df = od_labels_df.drop_duplicates(subset=['pano_id'])
    print('Number of od_labels after removing duplicates: {}'.format(len(od_labels_df)))

    # Convert PS labels to a GeoDataFrame
    ps_labels_df = gpd.GeoDataFrame.from_features(project_sidewalk_labels['features'])
    # Set crs to EPSG:4326
    ps_labels_df.crs = 'epsg:4326'
    print('Number of labels before filtering: {}'.format(len(ps_labels_df)))

    # Retain only Obstacle labels
    ps_labels_df = ps_labels_df[ps_labels_df['label_type'] == 'Obstacle']
    print('Number of labels after filtering: {}'.format(len(ps_labels_df)))

    # Filter ps_labels so that it only contains labels where "gsv_panorama_id" is in the list of pano_ids in od_labels
    ps_labels_df = ps_labels_df[ps_labels_df["gsv_panorama_id"].isin(od_labels_df["pano_id"])]
    print('Number of PS labels after filtering per panos: {}'.format(len(ps_labels_df)))

    # To make a meter buffer around the points, we need to 
    # convert the points to a different coordinate system (world-meter)
    ps_labels_df = ps_labels_df.to_crs("EPSG:32634")
    ps_labels_df.geometry = ps_labels_df.geometry.buffer(args.buffer_distance)

    # Convert back to WGS84
    ps_labels_df = ps_labels_df.to_crs("EPSG:4326")

    # Make it a GeoDataFrame
    gpd_ps_labels_df = gpd.GeoDataFrame(geometry=ps_labels_df.geometry)

    od_labels_df = od_labels_df.reset_index(drop=True)
    gpd_ps_labels_df = gpd_ps_labels_df.reset_index(drop=True)

    # join the two dataframes based on the points that intersect the buffered points in ps_labels_buffered
    joined_df = gpd.sjoin(od_labels_df, gpd_ps_labels_df, predicate='within')

    print('Number of predictions: ', len(od_labels_df))
    print('Number of PS labels: ', len(gpd_ps_labels_df))
    print('Number of intersected points: ', len(joined_df))

    # Calculate the number of correct predictions
    num_correct = len(joined_df)

    # Calculate precision and recall
    precision = num_correct / len(ps_labels_df)
    recall = num_correct / len(od_labels_df)

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)

    # Create a pandas dataframe to store the results
    metrics_df = pd.DataFrame({'F1 score': [f1], 'Precision': [precision], 'Recall': [recall]})

    # Print the results
    print(metrics_df)

    # Save the results to a csv file
    metrics_df.to_csv(os.path.join(directory, 'metrics.csv'), index=False)

    return ps_labels_df, od_labels_df

def map(ps_labels, od_labels, directory):
    # Map the PS labels and OD labels
    hmap = folium.Map(location=[52.3676, 4.90], zoom_start=12, tiles='cartodbpositron',)

    # Use apply to iterate over each polygon in the GeoDataFrame.
    # For the color, if od_labels are inside folium.GeoJson, color is green, else red
    ps_labels.apply(lambda row: folium.GeoJson(row.geometry.__geo_interface__, style_function=lambda x: {'color': 'red'}).add_to(hmap), axis=1)

    # Use apply to iterate over each polygon in the GeoDataFrame again
    ps_labels.apply(lambda row: folium.CircleMarker(location=row.geometry.centroid.coords[0][::-1], radius=2, color='red').add_to(hmap), axis=1)    
    # Add od_labels_df to the map
    od_labels.apply(lambda row:folium.CircleMarker(location=[row["lat"], row["lon"]], radius=2, color='blue').add_to(hmap), axis=1)

    # Save the hmap as a high resolution image
    hmap_path = os.path.join(directory, 'heatmap' + f'_{args.buffer_distance}m' + '.html')
    hmap.save(hmap_path)
    print(f'Map saved in {hmap_path}.')

def main(args):
    # Replace everything that is not a character with an underscore in neighbourhood string, and make it lowercase
    args.neighbourhood = re.sub(r'[^a-zA-Z]', '_', args.neighbourhood).lower()

    args.input_dir = os.path.join(args.input_dir, args.neighbourhood)
    args.input_dir = args.input_dir + '_' + args.quality
    args.input_dir = os.path.join(args.input_dir, 'backprojected')

    # Create the output directory if it doesn't exist
    directory = os.path.join(os.path.dirname(args.input_dir), 'evaluation')
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    ps_labels, od_labels = evaluate(args, directory)
    map(ps_labels, od_labels, directory)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--input_dir', type=str, default = 'res/dataset')
    parser.add_argument('--neighbourhood', type=str, default='osdorp')
    parser.add_argument('--quality', type=str, default='full')
    parser.add_argument('--buffer_distance', type=float, default=1.0, help='Buffer distance in meters')
    
    args = parser.parse_args()
    
    main(args)