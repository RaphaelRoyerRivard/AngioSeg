import torch
from torch_geometric.data import Data
import numpy as np
from vesselanalysis import get_graph_data_from_raw_image_and_segmentation

if __name__ == '__main__':
    images_path = r"C:\Users\Raphael\Pictures\skeleton\1667408_RCA_0_0"
    raw_image_path = images_path + r"\1667408_RCA_0_0.tif"
    segmented_image_path = images_path + r"\1667408_RCA_0_0_seg.tif"
    edges, x, edge_attr, edge_ids = get_graph_data_from_raw_image_and_segmentation(raw_image_path, segmented_image_path, oriented=True)
    edge_index = torch.tensor(edges, dtype=torch.long)
    x = np.array(x)
    x[:, 2] /= np.max(x[:, 2], axis=0)  # normalize vessel width
    edge_attr = np.array(edge_attr)
    edge_attr[:, 0:2] /= np.max(edge_attr[:, 0:2], axis=0)  # normalize vessel length and average vessel width
    x = torch.tensor(x, dtype=torch.float)  # position_x, position_y, vessel_width, pixel_intensity, is_ostium
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # vessel_length, average_vessel_width, average_pixel_intensity
    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr.t().contiguous())
    print("edge_index", edge_index)
    print("x", x)
    print("edge_attr", edge_attr)
    print(data)
