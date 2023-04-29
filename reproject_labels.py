import argparse
import os
import numpy as np
import pandas as pd
from lib.py360convert import py360convert
from tqdm import tqdm

def compute_label_coordinates(dataframe, shape):

    labels = dataframe
    scaled_pano_x = []
    scaled_pano_y = []

    # For each label, find its coordinates in the panorama
    for index, row in labels.iterrows():
        image_width = row['pano_width']
        image_height = row['pano_height']
        sv_image_x = row['pano_x']
        sv_image_y = row['pano_y']

        sv_pano_x = int(sv_image_x*shape[1]/image_width)
        sv_pano_y = int(sv_image_y*shape[0]/image_height)

        scaled_pano_x.append(sv_pano_x)
        scaled_pano_y.append(sv_pano_y)
    
    # Add the new columns to the dataframe
    labels['scaled_pano_x'] = scaled_pano_x
    labels['scaled_pano_y'] = scaled_pano_y

    return labels

def reorient_point(point, img_size, heading):
    # We select a tolerance of 0.1 degrees so that 
    # the point is not reoriented if the heading is close to 0
    tolerance = 0.1
    if abs(heading) <= tolerance:
        return point

    else:
        # Reshift point according to heading
        shift = heading / 360
        pixel_split = int(img_size[1] * shift)
        
        y, x = point
        new_x = (x - pixel_split) % img_size[1]

        reoriented_point = (y, new_x)
        
        return reoriented_point

def reorient_labels(args, dataframe):
    print('Reorienting labels...')
    # Retrieve pano.csv with headings information
    panos_df = pd.read_csv(args.panos_csv_path)

    # panos_df contains the pano_id and the heading information, while the labels in dataframe contain
    # the pano id in the 'gsv_panorama_id' column. Find the heading for each label by matching the pano_id
    # and use it as arguments in reorient_label(heading)
    gt_points = []
    for index, row in tqdm(dataframe.iterrows()):
        pano_id = row['gsv_panorama_id']
        filtered_panos_df = panos_df.loc[panos_df['pano_id'] == pano_id]

        # Check the case in which the pano could not be downloaded due to API failure
        if not filtered_panos_df.empty:
            heading = filtered_panos_df['heading'].values[0]
            gt_points.append(reorient_point((row['scaled_pano_y'], row['scaled_pano_x']),
                                            (args.pano_height, args.pano_width), heading))
        else:
            gt_points.append((-1, -1))

    # Add the new columns to the dataframe
    dataframe['reoriented_pano_x'] = [point[1] for point in gt_points]
    dataframe['reoriented_pano_y'] = [point[0] for point in gt_points]

    print('Done!')
    return dataframe

def reproject_labels(args, dataframe):
    print('Reprojecting labels...')
    def reproject_point(y, x, pano_height, pano_width, face_w=256):
        # Step 1: Convert (y, x) to UV coordinates
        coor = np.array([x, y])
        uv = py360convert.utils.coor2uv(coor, pano_height, pano_width)

        # Step 2: Convert UV coordinates to XYZ coordinates
        xyz = py360convert.utils.uv2unitxyz(uv)

        # Step 3: Determine the face and coordinates within the face
        face_type = py360convert.utils.equirect_facetype(pano_height, pano_width)
        face_idx = face_type[y, x]

        cube_faces_xyz = py360convert.utils.xyzcube(face_w)
        face_xyz = cube_faces_xyz[:, face_idx * face_w:(face_idx + 1) * face_w, :]

        uv_face = py360convert.utils.xyz2uv(face_xyz)
        coor_xy_face = py360convert.utils.uv2coor(uv_face, pano_height, pano_width)

        y_proj, x_proj = np.unravel_index(
            np.argmin(np.linalg.norm(coor_xy_face - coor, axis=-1)), (face_w, face_w)
        )

        # Step 4: Return the face index and the new coordinates
        return face_idx, (y_proj, x_proj)
    
    # face_idx, (y_proj,x_project)
    gt_points = []
    for index, row in tqdm(dataframe.iterrows()):
        if row['reoriented_pano_x'] == -1:
            gt_points.append((-1, (-1, -1)))
        else:
            gt_points.append(reproject_point(row['reoriented_pano_y'], row['reoriented_pano_x'], \
                                         args.pano_height, args.pano_width, args.size))

    # Add the new columns to the dataframe
    dataframe['face_idx'] = [point[0] for point in gt_points]
    dataframe['reprojected_pano_x'] = [point[1][1] for point in gt_points]
    dataframe['reprojected_pano_y'] = [point[1][0] for point in gt_points]

    print('Done!')
    return dataframe

def main(args):
    # Load the csv file containing the labels using pandas
    labels_df = pd.read_csv(args.labels_csv_path)

    # Add the scaled coordinates to the dataframe
    scaled_labels_df = compute_label_coordinates(labels_df, (args.pano_height, args.pano_width))

    # 1. Reorient the labels (add the columns reoriented_pano_x and reoriented_pano_y)
    reoriented_labels_df = reorient_labels(args, scaled_labels_df)

    # 2. Reproject the labels (add the columns face_idx, reprojected_pano_x and reprojected_pano_y)
    reprojected_labels_df = reproject_labels(args, reoriented_labels_df)

    # 2.1. Remove the labels that could not be reprojected (face_idx = -1)
    reprojected_labels_df = reprojected_labels_df.loc[reprojected_labels_df['face_idx'] != -1]

    # 3. Save the dataframe as a csv file
    reprojected_labels_df.to_csv(args.labels_csv_path[:-4] + f'_{args.size}_reprojected.csv', index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--panos_csv_path', type=str, default = 'res/dataset/panos.csv')
    parser.add_argument('--labels_csv_path', type=str, default = 'res/dataset/labels.csv')
    parser.add_argument('--pano_width', type=int, default = 2048)
    parser.add_argument('--pano_height', type=int, default = 1024)
    parser.add_argument('--size', type=int, default = 512)
    
    args = parser.parse_args()
    
    main(args)