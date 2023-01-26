''' 
Collect street level panoramic images of Amsterdam from the data portal of the City of Amsterdam.
The script can be used to select a specific area of interest (e.g. neighbourhood) and download all images within that area.

@author: Andrea Lombardo
'''

import os
import argparse
import requests
import pandas as pd

def convert_bbox(bbox):
    '''Convert the bounding box coordinates from string to quadruple of floats'''
    bounds = tuple(map(float, bbox.split(',')))
    return bounds

def get_pano_count(bbox, page_size, timestamp_after):
    '''Method to get the number of panoramas in the area of interest.
    This is useful to know how many pages we need to loop through to get all the panoramas.'''
    
    bounds = convert_bbox(bbox)
    
    # API url
    url = (
    f"https://api.data.amsterdam.nl/panorama/panoramas/?format=json&page_size={page_size}" 
    f"&page=1&srid=4326&bbox={bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}&timestamp_after={timestamp_after}" 
    )

    test_api = requests.get(url).json()
    
    return test_api['count']

def get_pano_links(bbox, page_size, page, timestamp_after):
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
     The link to get our panorama is in '_links' -> 'equirectangular_full' -> 'href'
    '''

    # Get the number of panoramas in the area of interest and the bounding box coordinates
    count = get_pano_count(bbox, page_size, timestamp_after)
    print('Number of panoramas: ', count)
    
    # Bounding box coordinates for the area of interest
    bounds = convert_bbox(bbox)

    # Given the count of the panoramas, we can calculate the number of pages we need to loop through
    pages = count // page_size + 1
    print('Number of pages: ', pages)

    # For each page, collect all the links to the panoramas, their ids and their headings
    for page in range(1, pages+1):
        print('Collecting panos of page ', page)
        # API url
        url = (
        f"https://api.data.amsterdam.nl/panorama/panoramas/?format=json&page_size={page_size}" 
        f"&page={page}&srid=4326&bbox={bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}&timestamp_after={timestamp_after}"
        )

        test_api = requests.get(url).json()

        pano_ids = [pano['pano_id'] for pano in test_api['_embedded']['panoramas']]
        links = [pano['_links']['equirectangular_full']['href'] for pano in test_api['_embedded']['panoramas']]
        headings = [pano['heading'] for pano in test_api['_embedded']['panoramas']]
        print('Number of links: ', len(links))

    # Assert if the number of links is equal to the number of panoramas and headings
    assert len(links) == len(pano_ids)
    assert len(links) == len(headings)
    
    return links, pano_ids, headings

def save_panos(links, pano_ids, headings, dir, is_sample=True):
    '''Save the panoramas in the input directory and make a .csv file with the headings of the panoramas'''

    os.makedirs(dir, exist_ok=True)

    saved_count = 0
    for link, pano_id in zip(links, pano_ids):
        if is_sample:
            if saved_count == 1000:
                break
        r = requests.get(link, allow_redirects=True)
        if r.status_code == 200:
            open(f'{dir}/{pano_id}.jpg', 'wb').write(r.content)
            saved_count += 1
    print(f'Number of panoramas saved: {saved_count} out of {len(links) - saved_count}')

    # Save the panos with their headings to a csv file
    data = {'pano_id': pano_ids, 'heading': headings}
    # The headers of the csv are: pano_id, heading
    df = pd.DataFrame(data)
    df.to_csv(dir + '/panos.csv', index=False)

def main(args):
    '''Main function to collect the panoramas'''
    links, panos_id, headings = get_pano_links(args.bbox, args.page_size, args.page, args.timestamp_after)
    save_panos(links, panos_id, headings, args.output_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Default bounds: Osdorp
    parser.add_argument('--bbox', type=str, default='4.754844, 52.346254, 4.820438, 52.381408')
    parser.add_argument('--page_size', type=int, default=10000)
    parser.add_argument('--page', type=int, default=1)
    parser.add_argument('--timestamp_after', type=str, default='2021-01-01')    
    parser.add_argument('--output_dir', type=str, default = 'res/dataset')
    
    args = parser.parse_args()
    
    main(args)