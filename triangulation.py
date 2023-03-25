'''The script is a mix of code coming from 
https://github.com/Computer-Vision-Team-Amsterdam/Geolocalization_of_Street_Objects

NOTE: Currently, we do not establish approximate camera-to-object distances.
Therefore, only normalized locations (at 1m distance from camera) are used.'''

import requests
import json
import csv

from osgeo import ogr
from math import sqrt
from osgeo import osr
from pathlib import Path
from panorama.client import PanoramaClient
from panorama import models
import pycocotools.mask as mask_util

import time
from math import radians, cos, sin
import numpy as np
import json
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from imantics import Mask
from shapely.geometry import Polygon


def rd_to_wgs(coords):
    """
    Convert rijksdriehoekcoordinates into WGS84 cooridnates. Input parameters: x (float), y (float).
    """
    epsg28992 = osr.SpatialReference()
    epsg28992.ImportFromEPSG(28992)

    epsg4326 = osr.SpatialReference()
    epsg4326.ImportFromEPSG(4326)

    rd2latlon = osr.CoordinateTransformation(epsg28992, epsg4326)
    lonlatz = rd2latlon.TransformPoint(coords[0], coords[1])
    return lonlatz[:2]


def pixel_to_viewpoint(pixel, image_width):
    """
    Convert width in pixels to viewpoint in degrees
    """
    return 360 * pixel / image_width

def viewpoint_to_pixels(viewpoint, image_width):
    """
    Convert viewpoint in degrees to width in pixels
    """
    return viewpoint / 360 * image_width

def euclidean_distance(x_1, y_1, x_2, y_2):
    """
    Calculate the Euclidean distance between two vectors
    """
    return sqrt(((x_1 - x_2)**2) + ((y_1 - y_2)**2))

def get_pano_location_ps(pano_id):
    """
    Get the initial location coordinates of a panoramic image
    and convert it from WGS84 (EPSG:4326) to Rijksdriehoek (EPSG:28992)

    Coordinates retrieved from panos_coords.csv in res/dataset_PS/neighbourhood_quality
    """
    # Open .csv file with the IDs and their latitudes and longitudes, find the coordinates of the panorama
    with open(f'res/dataset_PS/centrum_west_small/panos_coords.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row['pano_id'] == pano_id:
                geom = [float(row['lat']), float(row['lng'])]
                break
    return geom


def get_pano_location(pano_id, ps_labels=False):
    """
    Get the initial location coordinates of a panoramic image
    and convert it from WGS84 (EPSG:4326) to Rijksdriehoek (EPSG:28992)

    Amsterdam API description: https://api.data.amsterdam.nl/api/
    """
    if ps_labels:
        geom = get_pano_location_ps(pano_id)
        return geom
    else:
        pano_url = f"https://api.data.amsterdam.nl/panorama/panoramas/{pano_id}/"

        try:
            response = requests.get(pano_url)
            pano_data = json.loads(response.content)
        except requests.exceptions.RequestException:
            print('HTTP Request failed. Aborting.')
            return
        
        # Get location coordinates
        geom = pano_data['geometry']['coordinates']

    # WGS84 to RD conversion
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(max(geom), min(geom))
    #
    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)

    target = osr.SpatialReference()
    target.ImportFromEPSG(28992)

    transform = osr.CoordinateTransformation(source, target)
    point.Transform(transform)

    return [point.GetX(), point.GetY()]


def get_panos_from_points_of_interest(point_of_interest, timestamp_before, timestamp_after):
    """
    Get the panoramas that are closest to the points of interest
    Takes as input a csv file with lon, lat coordinates outputs images in the output folder
    """
    panoramas = []
    with open(point_of_interest, 'r') as file:
        reader = csv.reader(file)
        next(reader) #skip first line
        for row in reader:
            location = models.LocationQuery(
                latitude=float(row[0]), longitude=float(row[1]), radius=25
            )
            query_result: models.PagedPanoramasResponse = PanoramaClient.list_panoramas(
                location=location,
                timestamp_after=timestamp_after,
                timestamp_before=timestamp_before)
            panoramas.append(query_result.panoramas[0])
    return panoramas

"""
This file contains a refactored Python implementation of the MRF-based
triangulation procedure to estimate object geolocations, introduced in
"Automatic Discovery and Geotagging of Objects from Street View Imagery"
by V. A. Krylov, E. Kenny, R. Dahyot.
https://arxiv.org/abs/1708.08417

The approach is not restricted to any specific type of object as long as
detected objects are compact enough to be described by a single geotag.

NOTE: Currently, we do not establish approximate camera-to-object distances.
Therefore, only normalized locations (at 1m distance from camera) are used.
"""



MAX_DST_CAM_OBJECT = 15  # Max distance from camera to objects (in meters)
MAX_CLUSTER_SIZE = 1  # Maximal size of clusters employed (in meters)

# MRF optimization parameters
ICM_ITERATIONS = 15  # Number of iterations for ICM
DEPTH_WEIGHT = 0.2  # weight alpha in Eq.(4)
OBJECT_MULTIVIEW = 0.2  # weight beta in  Eq.(4)
STANDALONE_PRICE = max(1 - DEPTH_WEIGHT - OBJECT_MULTIVIEW,
                      0)  # weight (1-alpha-beta) in Eq. (4)


def intersection_point(object_1, object_2):
    """
    Calculating the intersection point between two lines
    Specified each by camera location and depth-estimated object location
    """

    # The camera locations (index 4 and 5) and the object locations (index 0 and 1)
    a_1 = object_1[0] - object_1[4]
    b_1 = object_2[0] - object_2[4]
    c_1 = object_2[4] - object_1[4]

    a_2 = object_1[1] - object_1[5]
    b_2 = object_2[1] - object_2[5]
    c_2 = object_2[5] - object_1[5]

    if a_2 * b_1 - b_2 * a_1:
        y = (a_1 * c_2 - a_2 * c_1) / (a_2 * b_1 - b_2 * a_1)
    else:
        return -1, -1, 0, 0
    if a_1 != 0:
        x = (b_1 * y + c_1) / a_1
    else:
        x = (b_2 * y + c_2) / a_2

    if x < 0 or y < 0:
        return -2, -2, 0, 0
    if x > MAX_DST_CAM_OBJECT or y > MAX_DST_CAM_OBJECT:
        return -3, -3, 0, 0

    # Calculate the intersection point
    x_intersect, y_intersect = a_1 * x + object_1[4], a_2 * x + object_1[5]

    return x, y, x_intersect, y_intersect


def calc_energy(object_dst, objects_base, objects_connectivity, object_1):
    """
    Calculate the MRF energy of an intersection
    """
    inters = np.count_nonzero(objects_connectivity[object_1, :])
    if inters == 0:
        return STANDALONE_PRICE
    energy = 0
    dpthmin, dpthmax = 1000, 0
    for i in range(len(objects_base)):
        if objects_connectivity[object_1, i]:
            dpth_temp = DEPTH_WEIGHT * abs(object_dst[object_1, i]
                                        - objects_base[object_1][3])
            energy += dpth_temp
            dpth = object_dst[object_1, i]
            if dpth < dpthmin:
                dpthmin = dpth
            if dpth > dpthmax:
                dpthmax = dpth
    return energy + OBJECT_MULTIVIEW * (dpthmax - dpthmin)


def avg_object_location(intersects, objects_connectivity, object_1):
    """
    Calculate the averaged object location (used after clustering)
    """
    res = np.zeros(2)
    cnt = 0
    for i in range(intersects.shape[0]):
        if objects_connectivity[object_1, i]:
            res[:] += intersects[object_1, i, :]
            cnt += 1
    if cnt:
        return res / cnt
    return res


def hierarchical_cluster(intersects, max_intra_degree_dst):
    """
    Hierarchical clustering
    """
    Z = linkage(np.asarray(intersects))
    clusters = fcluster(Z, max_intra_degree_dst, criterion="distance") - 1
    num_clusters = max(clusters) + 1
    cluster_intersections = np.zeros((num_clusters, 3))
    for i in range(len(intersects)):
        cluster_intersections[clusters[i], 0] += intersects[i][0]
        cluster_intersections[clusters[i], 1] += intersects[i][1]
        cluster_intersections[clusters[i], 2] += 1
    return cluster_intersections


def read_inputfile(input_file, ps_labels=False):
    """
    Read the input CSV file that defines a detected object by four values of
    type float: camera location RD-coordinates (X, Y), viewpoint from north
    clockwise in degrees towards the object in the panoramic image and the
    depth estimate. The latter may be omitted or set to zero.
    """
    objects_base = []

    with open(input_file) as f:
        for detected_instance in tqdm(json.load(f)):
            pano_id = detected_instance['pano_id'].replace(".jpg", "")
            img_width = detected_instance['segmentation']['size'][1]
            #segmentation_mask = detected_instance['segmentation']['counts']
            segmentation_mask = mask_util.decode(detected_instance['segmentation'])
            polygons = Mask(segmentation_mask).polygons()
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
            # Code added to differentiate if we are evaluating the model with Project Sidewalk
            # labels or not
            if (ps_labels):
                x, y = get_pano_location_ps(pano_id)
            else:
                x, y = get_pano_location(pano_id)
            viewpoint_to_object = pixel_to_viewpoint(center_x, img_width)
            # We assume the depth is 3 meters (2 from camera to sidewalk, 1 from sidewalk to object)
            depth = 3 # TODO: ADD DEPTH ESTIMATION

            if depth <= 0:
                depth = 3

            # Calculating the object locations using
            # camera location + viewpoint_to_object + depth_estimate
            br1 = radians(180 + viewpoint_to_object)
            x_object = x + (depth * sin(br1) * 640 / 256) # depth-based locations
            y_object = y + (depth * cos(br1) * 640 / 256)
            # Normalized locations (at 1m distance from camera)
            # In our use case, we 
            x_object_norm = x + (1.0 * sin(br1) * 640 / 256)
            y_object_norm = y + (1.0 * cos(br1) * 640 / 256)

            if pano_id == "XvagOYHRvAMjYlzwNg7rmw":
                print('Printing information about panorama XvagOYHRvAMjYlzwNg7rmw:\n')
                print('x: ', x)
                print('y: ', y)
                print('viewpoint_to_object: ', viewpoint_to_object)
                print('br1 (radians): ', br1)
                print('x_object: ', x_object)
                print('y_object: ', y_object)
                print('What we add to x: ', (depth * sin(br1) * 640 / 256))
                print('What we add to y: ', (depth * cos(br1) * 640 / 256))
                print('x_object_norm: ', x_object_norm)
                print('y_object_norm: ', y_object_norm)

            if (ps_labels):
                # Test if this works
                x_object_norm, y_object_norm = rd_to_wgs((y_object, x_object))

            if pano_id == 'XvagOYHRvAMjYlzwNg7rmw':
                print('x_object_norm: ', x_object_norm)
                print('y_object_norm: ', y_object_norm)

            

            objects_base.append((
                x_object_norm,
                y_object_norm,
                viewpoint_to_object, # not used
                depth,
                x,
                y,
                x_object,   # not used, see NOTE at the top
                y_object   # not used, see NOTE at the top
            ))

    print("All detected objects: {0:d}".format(len(objects_base)))
    return objects_base


def get_all_intersections(objects_base):
    """
    Get the RD-coordinates of the pairwise intersections
    """
    num_intersections = 0
    object_dst = np.zeros((len(objects_base), len(objects_base)))
    intersections = np.zeros((len(objects_base), len(objects_base), 2))

    # Maximum distance between the two camera locations observing the same object
    max_cam_dst = 1.5 * MAX_DST_CAM_OBJECT

    # Nested loop, looping over all initial locations of panoramic images,
    # checking which one are close to each other
    for i in range(len(objects_base)):
        # An update to the user
        if i % 1000 == 0 and i > 0:
            print("Parced {} object entries ({:.2f}%)".format(i, 100.
                                                              * i / len(objects_base)))

        # Set an error value when identical panoramic image
        object_dst[i, i] = -5

        # Only pairwise intersections
        for j in range(i + 1, len(objects_base)):
            # Calculate distance between two panoramic images in meters
            cam_dst = euclidean_distance(objects_base[i][5], objects_base[i][4],
                                            objects_base[j][5], objects_base[j][4])

            # Continue to next iteration if distance is less than 0.5m apart or too far
            if cam_dst < 0.5 or cam_dst > max_cam_dst: # NOTE maybe set this to 1m
                object_dst[i, j] = -4 # Set an error value
                object_dst[j, i] = -4
                continue

            # Get the distance to object and the RD-coordinates of the intersection
            object_dst[i, j], object_dst[j, i], intersections[i, j, 0], intersections[i, j, 1] = \
                                intersection_point(objects_base[i], objects_base[j])

            # The other way around is the same
            intersections[j, i, 0], intersections[j, i, 1] = \
                                            intersections[i, j, 0], intersections[i, j, 1]

            if object_dst[i, j] > 0:
                num_intersections += 1

    print("All admissible intersections: {0:d}".format(num_intersections))

    return object_dst, intersections


def mrf_energy_minimization(object_dst, objects_base):
    """
    The designed MRF model operates on an irregular grid that consists of all of the
    intersections in the previous step. Energy minimization is achieved with Iterative
    Conditional Modes (ICM).
    """

    objects_connectivity = np.zeros((len(objects_base),
                                    len(objects_base)), dtype=np.uint8)

    objects_connectivity_viable = np.zeros(len(objects_base),
                                                dtype=np.uint8)

    for i in range(len(objects_base)):
        objects_connectivity_viable[i] = \
            np.count_nonzero(object_dst[i, :] > 0)

    np.random.seed(int(100000.0 * time.time()) % 1000000000)
    chngcnt = 0
    for i in range(ICM_ITERATIONS * len(objects_base)):
        if (i + 1) % len(objects_base) == 0:
            print("Iteration #{}: accepted {} changes".format((i
                                                               + 1) / len(objects_base), chngcnt))
            chngcnt = 0
        test_objectect = np.random.randint(0, len(objects_base))
        # no pairing possible (standalone - )
        if objects_connectivity_viable[test_objectect] == 0:
            continue

        randnum = 1 + np.random.randint(0, objects_connectivity_viable[test_objectect])
        curcnt = 0
        for j in range(len(objects_base)):
            if object_dst[test_objectect, j] > 0:
                curcnt += 1
            if curcnt == randnum:
                # Test the object pair
                test_object_pair = j
                break

        energy_old = calc_energy(object_dst, objects_base,
                                     objects_connectivity, test_objectect)

        energy_old += calc_energy(object_dst, objects_base,
                                      objects_connectivity, test_object_pair)

        objects_connectivity[test_objectect, test_object_pair] = 1 \
            - objects_connectivity[test_objectect, test_object_pair]
        objects_connectivity[test_object_pair, test_objectect] = 1 \
            - objects_connectivity[test_object_pair, test_objectect]

        energy_new = calc_energy(object_dst, objects_base,
                                     objects_connectivity, test_objectect)
        energy_new += calc_energy(object_dst, objects_base,
                                      objects_connectivity, test_object_pair)

        if energy_new <= energy_old:
            chngcnt += 1
            continue

        # revert to the old configuration
        objects_connectivity[test_objectect, test_object_pair] = 1 \
            - objects_connectivity[test_objectect, test_object_pair]
        objects_connectivity[test_object_pair, test_objectect] = 1 \
            - objects_connectivity[test_object_pair, test_objectect]

    return objects_connectivity


def clustering(objects_base, objects_connectivity, intersects):
    """
    To obtain the final object configuration we perform clustering of MRF output in
    order to merge groups of object instances that describe the same physical object.
    """
    d45 = 0.707 * MAX_CLUSTER_SIZE * 640 / 256 # TODO explain
    ax, ay = objects_base[0][0] + d45, objects_base[0][1] + d45
    max_intra_degree_dst = euclidean_distance(ax, ay, objects_base[0][0], objects_base[0][1])

    icm_intersect = []
    for i in range(len(objects_base)):
        res = avg_object_location(intersects, objects_connectivity, i)
        if res[0]:
            icm_intersect.append((res[0], res[1]))

    print("ICM inrersections: {0:d}".format(len(icm_intersect)))

    # Merge positive intersections that are likely to describe the same object.
    cluster_intersections = hierarchical_cluster(icm_intersect, max_intra_degree_dst)

    return cluster_intersections


def convert_intersections_to_wgs_coords(intersections):
    """
    Converts all points of interest from Rijksdriekhoek to wgs84 coordinates
    """
    for i in range(len(intersections)):
        intersections[i][:2] = rd_to_wgs(intersections[i][:2] / intersections[i][2])
    return intersections


def write_output(output_file_name, clustered_intersections):
    """
    Write clustered intersections (a 2D list of floats) to an output file.
    """
    num_clusters = clustered_intersections.shape[0]

    print("Number of output ICM clusters: {0:d}".format(num_clusters))

    with open(output_file_name, "w") as inter:
        inter.write("lat,lon\n")
        for i in range(num_clusters):
            inter.write("{0:f},{1:f}\n".format(clustered_intersections[i,0] , clustered_intersections[i, 1]))

    print(f"Done writing cluster intersections to the file: {output_file_name}.")


def triangulate(coco_file, ps_labels=False):
    """
    Finds all the points of interest from provided predictions in COCO format on panorama images,
    outputs a csv file that lists all identified objects with their corresponding location
    """
    start = time.time()

    # Step 1: Read data from the input COCO file
    objects_base = read_inputfile(coco_file, ps_labels)

    # Step 2: Get the location of intersections
    object_dst, intersections = get_all_intersections(objects_base)

    # Step 3: MRF-based optimization approach
    objects_connectivity = mrf_energy_minimization(object_dst, objects_base)

    # Step 4: Cluster intersections
    clustered_intersections = clustering(objects_base, objects_connectivity, intersections)

    clustered_intersections = convert_intersections_to_wgs_coords(clustered_intersections)

    print("Elapsed total time: {0:.2f} seconds.".format(time.time() - start))

    return clustered_intersections