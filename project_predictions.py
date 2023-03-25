'''Script that runs after backproject.py. It performs the following actions:
1. Select panos that have obstacles labels from Project Sidewalk
2. Perform triangulation of the masks
3, Visualize both PS and prediction points in the map
4. Calculate accuracy, precision, recall'''

import sys
import os
import json
import argparse
import re
import numpy as np
import requests
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tqdm as tqdm

import cv2

import triangulation as tr

import pycocotools.mask as mask_util
from shapely.geometry import Polygon
from imantics import Mask
import math

def api_call():
    '''Get labels from Project Sidewalk'''

    base_url = 'https://sidewalk-amsterdam.cs.washington.edu/v2/access/attributesWithLabels?' + \
    'lat1={}&lng1={}&lat2={}&lng2={}' 
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

    # Only keep labels from "Centrum-West" neighbourhood, and print length before and after
    ps_labels_df = gpd.GeoDataFrame.from_features(project_sidewalk_labels['features'])
    print('Number of labels before filtering: {}'.format(len(ps_labels_df)))

    return ps_labels_df


def filter_panos(ps_labels_df):
    # Filter out labels that are not obstacles
    ps_labels_df = ps_labels_df[ps_labels_df['label_type'] == 'Obstacle']
    print('Number of labels after filtering: {}'.format(len(ps_labels_df)))

    # Extract gsv_panorama_id column from the dataframe and collect them in a list
    pano_ids = ps_labels_df['gsv_panorama_id'].tolist()

    return ps_labels_df, pano_ids

def FOV_calculation(pano_path):
    '''We assume that the camera used in the Google Street View car has a full-frame sensor (36mm x 24mm), 
    so we can estimate the sensor size in meters using the pixel size and the resolution of the image. 
    We assume a pixel size of 1.4 micrometers and a resolution of 16384 x 8192. Thus, the sensor size
    is 44.8mm x 28.16mm.
    
    To estimate the distance camera-object, we assume that the object will always be on the sidewalk,
    and the distance camera-sidewalk is 
    
    We assume the camera orientation is horizontal, so we use the horizontal FOV.'''
    SENSOR_WIDTH = 0.0448  # width of sensor in meters
    SENSOR_HEIGHT = 0.02816  # height of sensor in meters
    CAMERA_HEIGHT = 2 # height of camera in meters
    DISTANCE_TO_SIDEWALK = 2.5 # distance in meters

    img = cv2.imread(pano_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Apply Hough transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    # Calculate vanishing points
    vanishing_points = cv2.computeCorrespondEpilines(lines, 1, np.array([]))

    # Calculate FOV
    distance_to_object = 3 # distance in meters
    focal_length = (SENSOR_WIDTH / 2) / np.tan(np.arctan2(vanishing_points[0][0][1], vanishing_points[1][0][1]) / 2)
    fov = 2 * np.arctan2(SENSOR_WIDTH / 2, focal_length)
    fov_degrees = np.rad2deg(fov)

    return fov_degrees

def estimate_coords(args, input_file):
    '''The following script estimates the coordinates of the object masks in the panoramic image
    by estimating the FOV, assuming a fixed distance object-camera, and then '''
    folder_path = os.path.join(os.path.dirname(args.input_dir), 'reoriented')



    with open(input_file) as f:
        for detected_instance in tqdm(json.load(f)):
            pano_id = detected_instance['pano_id']
            pano_path = os.path.join(folder_path, pano_id)

def final_projection(input_file):
    
    object_info = []
    # Known parameters
    '''- Assumptions: camera used on the Google Street View car rooftop has a spherical
    lens and captures a full 360-degree panoramic image. 
    11-lens camera system that captures images with a resolution of 16384x8192 pixels. 
    Since the images are panoramic and cover a full 360-degree field of view horizontally, 
    we can assume that the horizontal field of view is 360 degrees.

    - Pixel Size = Sensor Size / Number of Pixels
    - Sensor Size = (Horizontal Field of View) / (Number of Pixels in the Horizontal Direction)
    - Sensor Size = 360 / 16384 = 0.022 degrees/pixel
    - Pixel Size = Sensor Size / Number of Pixels = 0.022 / 16384 = 1.34 x 10^-6 degrees/pixel

    The pixel size in meters depends on the physical size of the camera sensor and the focal length of the lens.

    - Angular Resolution = Pixel Size / Focal Length

    Convert the angular resolution to linear resolution:
    
    - Linear Resolution = (2 x Distance x tan(Angular Resolution/2))

    where Distance is the distance between the camera and the object being imaged.

    Assuming camera height of about 3 meters above the ground and using
    an average focal length of 35mm for the camera lens, we can estimate the pixel size in meters as follows:

    Angular Resolution = (1.34 x 10^-6 degrees/pixel) / (35mm/1000mm) = 3.83 x 10^-8 radians/pixel
    Linear Resolution = (2 x 3 meters x tan(3.83 x 10^-8/2)) = 2.63 x 10^-5 meters/pixel

    Pixel size is approximately 2.63 x 10^-5 meters/pixel.'''

    
    pixel_size = 0.000263 # in meters (currently using 10^-4)
    distance = 3 # in meters

    with open(input_file) as file:
        for detected_instance in tqdm.tqdm(json.load(file)):
            pano_id = detected_instance['pano_id'].replace(".jpg", "")
            pano_width = detected_instance['segmentation']['size'][1]
            pano_height = detected_instance['segmentation']['size'][0]
            #segmentation_mask = detected_instance['segmentation']['counts']
            segmentation_mask = mask_util.decode(detected_instance['segmentation'])
            mask = Mask(segmentation_mask)
            polygons = mask.polygons()
            areas = []
            centroids = []
            for polygon_points in polygons.points:
                if len(polygon_points) < 3:
                    continue
                shapely_polygon = Polygon(polygon_points)
                areas.append(shapely_polygon.area)
                centroids.append(list(shapely_polygon.centroid.coords[0]))
            try:
                center_x = sum([areas[i] * centroids[i][0] for i in range(len(areas))]) / sum(areas)
                center_y = sum([areas[i] * centroids[i][1] for i in range(len(areas))]) / sum(areas)
            except ZeroDivisionError:
                continue

            # Initialize variables to hold the maximum width and height
            max_width = 0
            max_height = 0

            # Compute the bounding box for each polygon and update the maximum width and height
            for polygon_points in polygons.points:
                if len(polygon_points) < 3:
                    continue
                # Compute the bounding box of the contour
                x_min, y_min, x_max, y_max = Polygon(polygon_points).bounds

                # Compute the width and height of the bounding box
                width = x_max - x_min
                height = y_max - y_min

                # Update the maximum width and height
                max_width = max(max_width, width)
                max_height = max(max_height, height)

            #print(f"Mask width: {max_width}, mask height: {max_height}")

            # Get the size of the segmentation mask
            object_width = max_width
            object_height = max_height
            #object_width = max([point[0] for point in polygons.points[0]]) - min([point[0] for point in polygons.points[0]])
            #object_height = max([point[1] for point in polygons.points[0]]) - min([point[1] for point in polygons.points[0]])
            #print('Object width: {}, Object height: {}'.format(object_width, object_height))
            pano_lat, pano_long = tr.get_pano_location_ps(pano_id)
            #print('Latitude: {}, Longitude: {}'.format(pano_lat, pano_long))

            # Calculate field of view and object size
            fov = 2 * math.atan(pixel_size / (2 * distance))
            object_size = 2 * distance * math.tan(fov/2) * math.sqrt(object_width**2 + object_height**2)

            # Calculate the distance between the center of the object mask and the center of the panoramic image in pixels.
            dx = pano_width/2 - center_x
            dy = pano_height/2 - center_y
            distance_px = math.sqrt(dx**2 + dy**2)

            # Calculate the horizontal and vertical angles between the center of the object mask and the center of the panoramic image using the field of view, object size, and the distance calculated in steps 1 and 3.
            horizontal_angle = math.atan((dx * pixel_size + object_size/2) / (distance_px * pano_width)) # in radians
            vertical_angle = math.atan((dy * pixel_size + object_size/2) / (distance_px * pano_height)) # in radians

            # Calculate the latitude and longitude of the center of the object mask using the angles calculated in step 4 and the lat/long coordinates of the panoramic image.
            lat = pano_lat + math.degrees(vertical_angle)
            long = pano_long + math.degrees(horizontal_angle) / math.cos(math.radians(pano_lat))

            # Add the lat and long to the object_info list
            object_info.append([pano_id, lat, long])

    return object_info

def project(args, panos, output_file):

    input_file = os.path.join(args.input_dir, "input_coco_format.json")

    with open(input_file, 'r') as f:
            data = json.load(f)

    # Print number of panos
    print('Number of panos in the JSON file: {}'.format(len(data)))

    if args.pslabels:
        print('-- Using Project Sidewalk labels --')
        # input_file is a .json containing, for each pano, the key "pano_id" where the ID of the pano
        # is contained, and "segmentation". We are interested in only "pano_id".
        # We want to filter out panos that don't have obstacle labels.
        # To do so, we use "panos", which are exactly the panos that contain obstacle labels
        input_file_filtered = os.path.join(args.input_dir, "input_coco_format_filtered.json")

        # Print number of panos in panos
        print('Number of panos in panos: {}'.format(len(panos)))

        # Filter the data dictionary by IDs
        filtered_data = [elem for elem in data if elem['pano_id'] in panos]

        # Print the number of panos after filtering
        print('Number of panos in the JSON file after filtering: {}'.format(len(filtered_data)))


        # Write the filtered data to a new JSON file
        with open(input_file_filtered, 'w') as f:
            json.dump(filtered_data, f)
                
        object_info = final_projection(input_file_filtered)
        '''#cluster_intersections = tr.triangulate(input_file, True)
        # We cannot use triangulate function because the main assumption that the object
        # will be visible from multiple panos is not true. Hence, we assume that the distance
        # between the camera and the object is fixed.
        object_info = tr.read_inputfile(input_file_filtered, True)

        # Print one example of object_info
        print(object_info[0])'''
        
        # Write the output to a CSV file. The output file will contain the pano_id, latitude, and longitude of each object.
        with open(output_file, "w") as inter:
            inter.write("pano_id,lat,lon\n")
            for pano_id, lat, lon in object_info:
                inter.write("{},{},{}\n".format(pano_id, lat, lon))

        print(f"Done writing object locations to the file: {output_file}.")

    else:
        print('-- Using predicted labels --')
        cluster_intersections = tr.triangulate(input_file, False)
    
        tr.write_output(output_file, cluster_intersections)

def main(args):
    # Replace everything that is not a character with an underscore in neighbourhood string, and make it lowercase
    args.neighbourhood = re.sub(r'[^a-zA-Z]', '_', args.neighbourhood).lower()

    args.input_dir = os.path.join(args.input_dir, args.neighbourhood)
    args.input_dir = args.input_dir + '_' + args.quality
    args.input_dir = os.path.join(args.input_dir, 'backprojected')
    
    ps_labels = api_call()
    ps_labels, panos = filter_panos(ps_labels)

    output_file = os.path.join(args.input_dir, "object_locations.csv")

    # Perform triangulation and project predictions to GSV coordinates
    project(args, panos, output_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--input_dir', type=str, default = 'res/dataset')
    parser.add_argument('--neighbourhood', type=str, default='osdorp')
    parser.add_argument('--quality', type=str, default='full')
    parser.add_argument('--pslabels', type=bool, default=False, action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    
    main(args)