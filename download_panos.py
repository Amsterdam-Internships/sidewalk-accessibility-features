''' 
Collect street level panoramic images of Amsterdam from the data portal of the City of Amsterdam.
The script can be used to select a specific area of interest (e.g. neighbourhood) and download all images within that area.

@author: Andrea Lombardo
'''

import subprocess
import os
import argparse
import requests
import pandas as pd
import re
import numpy as np

from tqdm import tqdm

# Make a dictionary with key = neighbourhood and value = bounding box
# List of neighbourhoods can be found here: 
# https://maps.amsterdam.nl/open_geodata/geojson_lnglat.php?KAARTLAAG=INDELING_WIJK&THEMA=gebiedsindeling
# Note: '' has been changed to 'Havens-West and Coenhaven/Minervahaven'.
neighbourhoods = {'Oud-West, De Baarsjes': '4.867406,52.36685,4.876809,52.37341', 
'Oud-Zuid': '4.848181, 52.342768,  4.857573, 52.356523',
'De Pijp, Rivierenbuurt': '4.886485, 52.352231,  4.906056, 52.360232', 
'Bijlmer-Centrum': '4.937623, 52.315024,  4.952864, 52.32814',
'Buitenveldert, Zuidas': '4.864804, 52.341396,  4.885726, 52.346019',
'Centrum-West': '4.87444, 52.364925,  4.90641, 52.388692', 
'Slotervaart': '4.818066, 52.344636,  4.834413, 52.357892', 
'De Aker, Sloten, Nieuw-Sloten': '4.791108, 52.326928,  4.847139, 52.351746', 
'Indische Buurt, Oostelijk Havengebied': '4.910594, 52.36655 ,  4.956709, 52.382571', 
'Oud-Oost': '4.904634, 52.348115,  4.916499, 52.360591', 
'IJburg, Zeeburgereiland': '4.952428, 52.341537,  5.012848, 52.382651',
'Watergraafsmeer': '4.912233, 52.339812,  4.940186, 52.355337', 
'Centrum-Oost': '4.895268, 52.365779,  4.913608, 52.380041',
'Noord-West': '4.85572 , 52.413905,  4.898766, 52.430679',
'Oud-Noord': '4.863633, 52.382155,  4.912935, 52.41827',
'Noord-Oost': '4.922311, 52.395617,  4.939214, 52.411013',
'Weesp, Driemond':'4.996119, 52.302458,  5.021545, 52.324513',
'Sloterdijk Nieuw-West': '4.75749 , 52.38412 ,  4.844813, 52.400283', 
'Geuzenveld, Slotermeer': '4.75896 , 52.369641,  4.812171, 52.384656',
'Osdorp': '4.754844, 52.351272,  4.807539, 52.381408',
'Bijlmer-Oost': '4.956755, 52.316597,  4.977702, 52.327759',
'Gaasperdam': '4.955402, 52.294323,  4.97872 , 52.307997',
'Havens-West and Coenhaven/Minervahaven': '4.728768, 52.391895,  4.863633, 52.431066',
'Westerpark': '4.844117, 52.385108,  4.864963, 52.395524',
'Bos en Lommer': '4.835337, 52.372192,  4.847418, 52.385103',
'Bijlmer-West': '4.929332, 52.278176,  4.971842, 52.319335'}

def convert_bbox(bbox):
    '''Convert the bounding box coordinates from string to quadruple of floats'''
    bounds = tuple(map(float, bbox.split(',')))
    return bounds

def get_pano_count(bbox, page_size, timestamp_after, projection):
    '''Method to get the number of panoramas in the area of interest.
    This is useful to know how many pages we need to loop through to get all the panoramas.'''
    
    bounds = convert_bbox(bbox)
    
    # API url
    url = (
    f"https://api.data.amsterdam.nl/panorama/panoramas/?format=json&page_size={page_size}" 
    f"&page=1&srid={projection}&bbox={bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}&timestamp_after={timestamp_after}" 
    )

    test_api = requests.get(url).json()
    
    return test_api['count']

def get_pano_links(args):
    ''' The structure of the json file is as follows:
     - links:
     -- href: link to the json file
     -- next (page)
     -- previous (page)
     - count: number of panoramas returned by the API
     - _embedded: the actual json with the panoramas
     -- panoramas: list of panoramas
     Each panorama has the following keys: 
     '_links', 'cubic_img_baseurl', 'cubic_img_pattern', 'geometry', 'pano_id', 'timestamp', 
     'filename', 'surface_type', 'mission_distance', 'mission_type', 'mission_year', 'tags', 'roll', 'pitch', 'heading'
     Depending on the needs, the link to get the panorama is in '_links' ->
     (8000x6000): 'equirectangular_full' -> 'href'
     (4000x2000): 'equirectangular_medium' -> 'href'
     (2000x1000): 'equirectangular_small' -> 'href'
    '''

    # Find the coordinates of the bounding box for the neighbourhood selected
    bbox = neighbourhoods[args.neighbourhood]
    print(f'Neighbourhood selected: {args.neighbourhood}')
    print(f'Bounding box of the neighbourhood: {bbox}')
    print(f'Quality of the image: {args.quality}')
    print(f'Timestamp after: {args.timestamp_after}')
    print(f'Projection of coordinates: {args.projection}')

    # Get the number of panoramas in the area of interest and the bounding box coordinates
    count = get_pano_count(bbox, args.page_size, args.timestamp_after, args.projection)
    print('Number of panoramas: ', count)
    
    # Bounding box coordinates for the area of interest
    bounds = convert_bbox(bbox)

    # Given the count of the panoramas, we can calculate the number of pages we need to loop through
    pages = count // args.page_size + 1
    print('Number of pages: ', pages)

    # For each page, collect all the links to the panoramas, their ids and their headings
    # Make empty lists of pano_ids, links, headings
    pano_ids = []
    links = []
    headings = []
    coords = []
    mission_years = []
    for page in range(1, pages+1):
        print('Collecting panos of page ', page)
        # API url
        url = (
        f"https://api.data.amsterdam.nl/panorama/panoramas/?format=json&page_size={args.page_size}" 
        f"&page={page}&srid=4326&bbox={bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}&timestamp_after={args.timestamp_after}"
        )

        test_api = requests.get(url).json()
        
        # Collect the links, pano_ids and headings
        for pano in test_api['_embedded']['panoramas']:
            # Skip the panoramas with surface_type = "W", which means
            # that the panorama is not taken from the street (the ground surface is water)
            if pano['surface_type'] == 'W':
                continue
            else:
                pano_ids.append(pano['pano_id'])
                links.append(pano['_links'][f'equirectangular_{args.quality}']['href'])
                headings.append(pano['heading'])
                coords.append(pano['geometry']['coordinates'])
                mission_years.append(pano['mission_year'])

    # Assert if the number of links is equal to the number of panoramas and headings
    assert len(links) == len(pano_ids)
    assert len(links) == len(headings)
    assert len(links) == len(coords)
    assert len(links) == len(mission_years)
    
    return links, pano_ids, headings, coords, mission_years

def save_panos(links, pano_ids, headings, coords, mission_years, args):
    '''Save the panoramas in the input directory and make a .csv file with the headings of the panoramas'''

    # Make a directory specific for the neighbourhood
    # Save the panoramas in the directory

    # Replace everything that is not a character with an underscore in neighbourhood string, and make it lowercase
    args.neighbourhood = re.sub(r'[^a-zA-Z]', '_', args.neighbourhood).lower()

    dir_path = os.path.join(args.output_dir, args.neighbourhood)
    dir_path = dir_path + '_' + args.quality
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    saved_count = 0

    # True if we want to test the code
    testing = False

    if (testing):
        print('Testing the code')
        # For testing, randomly sample 1000 images from links
        np.random.seed(42)
        links_test = np.random.choice(links, 1000, replace=False)
        # Find the indeces of the links_test in links, and use these indeces to find the corresponding pano_ids and headings
        indeces = [links.index(link) for link in links_test]
        pano_ids_test = [pano_ids[i] for i in indeces]
        headings_test = [headings[i] for i in indeces]
        coords_test = [coords[i] for i in indeces]
        mission_years_test = [mission_years[i] for i in indeces]
        
        print('Number of indices: ', len(indeces))
        print('Number of pano_tests: ', len(pano_ids_test))
        print('Number of mission_years_test: ', len(mission_years_test))

        # Sanity check: 
        # Find the index of the 10th link in links
        print('Index of the 10th link in links: ', links.index(links_test[10]))
        # Find the index of the 10th pano_id in pano_ids
        print('Index of the 10th pano_id in pano_ids: ', pano_ids.index(pano_ids_test[10]))
        # Find the index of the 10th heading in headings
        print('Index of the 10th heading in headings: ', headings.index(headings_test[10]))
        # Find the index of the 10th coords in coords
        print('Index of the 10th coords in coords: ', coords.index(coords_test[10]))
        # Find the index of the 10th mission_year in mission_years
        print('Index of the 10th mission_year in mission_years: ', mission_years.index(mission_years_test[10]))

        for link, pano_id in tqdm(zip(links_test, pano_ids_test)):
            r = requests.get(link, allow_redirects=True)
            if r.status_code == 200:
                open(f'{dir_path}/{pano_id}.jpg', 'wb').write(r.content)
                saved_count += 1
        print(f'Number of panoramas saved: {saved_count} out of {len(links)}')

        # Save the panos with their headings to a csv file
        data = {'pano_id': pano_ids_test, 'heading': headings_test, 'coords': coords_test,
                 'mission_year': mission_years_test}
    else:
        # Save the first 2000 images
        #links_2000 = links[:2000]
        #pano_ids_2000 = pano_ids[:2000]

        for link, pano_id in tqdm(zip(links, pano_ids)):
            r = requests.get(link, allow_redirects=True)
            if r.status_code == 200:
                open(f'{dir_path}/{pano_id}.jpg', 'wb').write(r.content)
                saved_count += 1
        print(f'Number of panoramas saved: {saved_count} out of {len(links)}')

        # Save the panos with their headings to a csv file
        data = {'pano_id': pano_ids, 'heading': headings, 'coords': coords, 'mission_year': mission_years}
    # The headers of the csv are: pano_id, heading, coords, mission_year
    df = pd.DataFrame(data)
    df.to_csv(dir_path + '/panos.csv', index=False)

def main(args):

    if args.ps:
        print('Downloading panoramas from Project Sidewalk API')
        sidewalk_server_domain = 'sidewalk-amsterdam.cs.washington.edu'
        storage_path = args.output_dir
        subprocess.run(["python", "lib/sidewalk-panorama-tools/DownloadRunner.py", \
                        sidewalk_server_domain, storage_path])
        

    else:
        print('Downloading panoramas from DataPunt API')
        '''Main function to collect the panoramas'''
        links, panos_id, headings, coords, mission_years = get_pano_links(args)
        save_panos(links, panos_id, headings, coords, mission_years, args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--neighbourhood', type=str, default='Osdorp')
    parser.add_argument('--page_size', type=int, default=10000)
    parser.add_argument('--timestamp_after', type=str, default='2023-01-01')  
    parser.add_argument('--quality', type=str, default='small')
    parser.add_argument('--output_dir', type=str, default = 'res/dataset')
    parser.add_argument('--projection', type=str, default='4326')
    parser.add_argument('--ps', type=bool, default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    main(args)