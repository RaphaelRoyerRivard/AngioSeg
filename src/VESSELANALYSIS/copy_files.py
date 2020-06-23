import os
import argparse
from shutil import copyfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default=None, help='Path of the folder in which we want to recursively find the files to copy.')
    parser.add_argument('--contains', default=None, help='Substring to search in the name of the files we want to copy.')
    parser.add_argument('--output_folder', default=None, help='Path of the folder we want to copy the files to.')
    args = parser.parse_args()

    input_folder = args.input_folder
    contains = args.contains
    output_folder = args.output_folder

    for path, subfolders, files in os.walk(input_folder):
        print(path)
        for filename in files:
            if contains not in filename:
                continue

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            copyfile(f"{path}\\{filename}", f"{output_folder}\\{filename}")
