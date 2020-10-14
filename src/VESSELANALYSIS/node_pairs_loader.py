import os
import numpy as np


def load_node_pairs_from_path(data_path):
    data = {}
    for path, subfolders, files in os.walk(data_path):
        for pair_file in files:
            pairs = np.load(f"{path}/{pair_file}")
            image_pair = pair_file.split("__")
            image_pair = [img_name.split("_overlap")[0] for img_name in image_pair]
            image_pair[1] = image_pair[-1].split(".npy")[0]  # in case there was no "_overlap" in the names, we want to get rid of the extension
            data[(image_pair[0], image_pair[1])] = np.reshape(pairs, (pairs.shape[0], 2, 2))
    return data


if __name__ == '__main__':
    data = load_node_pairs_from_path("../PAIRINGTOOL/pairs")
    print(data)
