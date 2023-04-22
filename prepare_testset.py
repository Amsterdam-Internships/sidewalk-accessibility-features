import argparse
import os
import shutil
from random import sample, seed
from tqdm import tqdm

def create_testset(input_dir1, input_dir2, size, reorient, random_seed):
    print(f"Creating testsets with {size} common items from {input_dir1} and {input_dir2}...")

    seed(random_seed)

    if reorient:
        common_files = set(filter(lambda x: x.endswith('.jpg'), \
                    os.listdir(input_dir1))).intersection(set(filter(lambda x: x.endswith('.jpg'), os.listdir(input_dir2))))
    else:
        common_files = set(os.listdir(input_dir1)).intersection(set(os.listdir(input_dir2)))

    if len(common_files) < size:
        print(f'Error. Not enough common items to satisfy the requested size. Scaling down to {len(common_files)}.')
        size = len(common_files)

    selected_files = sample(common_files, size)

    output_dir1 = f"{input_dir1}_testset"
    output_dir2 = f"{input_dir2}_testset"

    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)

    for item in tqdm(selected_files):
        if reorient:
            shutil.copy(os.path.join(input_dir1, item), os.path.join(output_dir1, item))
            shutil.copy(os.path.join(input_dir2, item), os.path.join(output_dir2, item))
        else:
            shutil.copytree(os.path.join(input_dir1, item), os.path.join(output_dir1, item))
            shutil.copytree(os.path.join(input_dir2, item), os.path.join(output_dir2, item))

    print(f"Testsets created successfully with {size} common items in {output_dir1} and {output_dir2}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare testset folders from input directories.")
    parser.add_argument("--input_dir1", type=str, required=True, help="Path to first input folder")
    parser.add_argument("--input_dir2", type=str, required=True, help="Path to second input folder")
    parser.add_argument("--size", type=int, required=True, help="Number of items to copy")
    parser.add_argument("--reorient", action="store_true", help="Copy common .jpg images with different naming conventions instead of subfolders")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random number generation to ensure consistency across runs")

    args = parser.parse_args()

    create_testset(args.input_dir1, args.input_dir2, args.size, args.reorient, args.seed)