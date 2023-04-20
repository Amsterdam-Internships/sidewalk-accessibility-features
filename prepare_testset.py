import argparse
import os
import shutil
from random import sample

def create_testset(input_dir1, input_dir2, size):
    common_subfolders = set(os.listdir(input_dir1)).intersection(set(os.listdir(input_dir2)))
    
    if len(common_subfolders) < size:
        raise ValueError("Not enough common subfolders to satisfy the requested size.")
    
    selected_subfolders = sample(common_subfolders, size)
    
    output_dir1 = f"{input_dir1}_testset"
    output_dir2 = f"{input_dir2}_testset"

    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)

    for subfolder in selected_subfolders:
        shutil.copytree(os.path.join(input_dir1, subfolder), os.path.join(output_dir1, subfolder))
        shutil.copytree(os.path.join(input_dir2, subfolder), os.path.join(output_dir2, subfolder))

    print(f"Testsets created successfully with {size} common subfolders in {output_dir1} and {output_dir2}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare testset folders from input directories.")
    parser.add_argument("--input_dir1", type=str, required=True, help="Path to first input folder")
    parser.add_argument("--input_dir2", type=str, required=True, help="Path to second input folder")
    parser.add_argument("--size", type=int, required=True, help="Number of subfolders to copy")

    args = parser.parse_args()

    create_testset(args.input_dir1, args.input_dir2, args.size)
