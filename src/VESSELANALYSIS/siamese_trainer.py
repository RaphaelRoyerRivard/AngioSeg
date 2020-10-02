import torch
from torch_geometric.data import Data
import numpy as np
from vesselanalysis import get_graph_data_from_raw_image_and_segmentation


def get_graph_data_from_images(raw_image_path, segmented_image_path, oriented):
    edges, x, edge_attr, edge_ids = get_graph_data_from_raw_image_and_segmentation(raw_image_path, segmented_image_path, oriented)

    x = np.array(x)
    x[:, 2] /= np.max(x[:, 2], axis=0)  # normalize vessel width
    edge_attr = np.array(edge_attr)
    print(edge_attr.shape)
    edge_attr[:, 0:2] /= np.max(edge_attr[:, 0:2], axis=0)  # normalize vessel length and average vessel width

    edge_index = torch.tensor(edges, dtype=torch.long)
    x = torch.tensor(x, dtype=torch.float)  # position_x, position_y, vessel_width, pixel_intensity, is_ostium
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # vessel_length, average_vessel_width, average_pixel_intensity

    # print("edge_index", edge_index)
    # print("x", x)
    # print("edge_attr", edge_attr)

    return Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)


if __name__ == '__main__':
    # TODO the edges should be doubled when oriented=False

    # Image 1
    images_path = r"C:\Users\Raphael\Pictures\skeleton\1933060_LCA_-30_-25_2"
    raw_image_path = images_path + r"\1933060_LCA_-30_-25_2.tif"
    segmented_image_path = images_path + r"\1933060_LCA_-30_-25_2_seg.tif"
    data = get_graph_data_from_images(raw_image_path, segmented_image_path, oriented=True)
    print(data)

    # Image 2
    images_path = r"C:\Users\Raphael\Pictures\skeleton\1933060_LCA_90_0_3"
    raw_image_path = images_path + r"\1933060_LCA_90_0_3.tif"
    segmented_image_path = images_path + r"\1933060_LCA_90_0_3_seg.tif"
    data = get_graph_data_from_images(raw_image_path, segmented_image_path, oriented=True)
    print(data)
    print(data.is_directed())
