import os


def clean_skeletons_folder(skeletons_folder):
    for path, subfolders, files in os.walk(skeletons_folder):
        print(path)
        folder = path.split("/")[-1].split("\\")[-1]
        for filename in files:
            if filename != folder + ".tif" and filename != folder + "_seg.tif":
                os.remove(f"{path}\\{filename}")


if __name__ == '__main__':
    clean_skeletons_folder(r"C:\Users\Raphael\Pictures\skeleton - Copy")
