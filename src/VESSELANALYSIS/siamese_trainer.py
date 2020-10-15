import torch
from torch_geometric.data import Data
import numpy as np
from vesselanalysis import get_graph_data_from_raw_image_and_segmentation
from node_pairs_loader import load_node_pairs_from_path


def get_graph_data_from_images(raw_image_path, segmented_image_path, oriented):
    edges, x, edge_attr, edge_ids = get_graph_data_from_raw_image_and_segmentation(raw_image_path, segmented_image_path, oriented)

    x = np.array(x)
    x[:, 2] /= np.max(x[:, 2], axis=0)  # normalize vessel width
    # TODO the edges should be doubled when oriented=False
    edge_attr = np.array(edge_attr)
    edge_attr[:, 0:2] /= np.max(edge_attr[:, 0:2], axis=0)  # normalize vessel length and average vessel width

    edge_index = torch.tensor(edges, dtype=torch.long)
    x = torch.tensor(x, dtype=torch.float)  # position_x, position_y, vessel_width, pixel_intensity, is_ostium
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # vessel_length, average_vessel_width, average_pixel_intensity

    # print("edge_index", edge_index)
    # print("x", x)
    # print("edge_attr", edge_attr)

    return Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)


def get_graph_data_from_numpy_graph_file(graph_file_path):
    graph_data = np.load(graph_file_path, allow_pickle=True)

    x = torch.tensor(graph_data[0], dtype=torch.float)
    # TODO the edges should be doubled when oriented=False
    edge_index = torch.tensor(graph_data[1], dtype=torch.long)
    edge_attr = torch.tensor(graph_data[2], dtype=torch.float)

    return Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)


def get_node_pairs(graph_data_a, graph_data_b, point_pairs):
    node_pairs = []
    for i, point_pair in enumerate(point_pairs):
        diff_vectors_a = graph_data_a.x[:, :2] - point_pair[0]
        distances_a = torch.sqrt(torch.sum(diff_vectors_a ** 2, dim=1))
        node_index_a = torch.argmin(distances_a)
        if distances_a[node_index_a] > 0.007:  # Around 7 pixels for a 1024 x 1024 image
            print(f"Skipped pair {i}, distance_a = {distances_a[node_index_a]}")
            continue

        diff_vectors_b = graph_data_b.x[:, :2] - point_pair[1]
        distances_b = torch.sqrt(torch.sum(diff_vectors_b ** 2, dim=1))
        node_index_b = torch.argmin(distances_b)
        if distances_b[node_index_b] > 0.007:  # Around 7 pixels for a 1024 x 1024 image
            print(f"Skipped pair {i}, distance_b = {distances_b[node_index_b]}")
            continue

        node_pairs.append((node_index_a, node_index_b))

    return torch.tensor(node_pairs)


if __name__ == '__main__':

    # View 1
    view_name = "1933060_LCA_-30_-25_2"
    images_path = f"C:/Users/Raphael/Pictures/skeleton/{view_name}"
    # raw_image_path = images_path + r"\1933060_LCA_-30_-25_2.tif"
    # segmented_image_path = images_path + r"\1933060_LCA_-30_-25_2_seg.tif"
    # data = get_graph_data_from_images(raw_image_path, segmented_image_path, oriented=True)
    graph_file_path = images_path + f"/{view_name}_graph_oriented.npy"
    data = get_graph_data_from_numpy_graph_file(graph_file_path)
    print(data)

    # View 2
    view_name2 = "1933060_LCA_90_0_3"
    images_path2 = f"C:/Users/Raphael/Pictures/skeleton/{view_name2}"
    # raw_image_path2 = images_path2 + r"\1933060_LCA_90_0_3.tif"
    # segmented_image_path2 = images_path2 + r"\1933060_LCA_90_0_3_seg.tif"
    # data2 = get_graph_data_from_images(raw_image_path2, segmented_image_path2, oriented=True)
    graph_file_path2 = images_path2 + f"/{view_name2}_graph_oriented.npy"
    data2 = get_graph_data_from_numpy_graph_file(graph_file_path2)
    print(data2)

    # Load hand made node pairs
    node_pairs_path = r".\src\PAIRINGTOOL\pairs"
    image_pairs = load_node_pairs_from_path(node_pairs_path)
    point_pairs = torch.tensor(image_pairs[(view_name, view_name2)], dtype=torch.float)

    # Associate pairs with nodes
    node_pairs = get_node_pairs(data, data2, point_pairs)
    print(node_pairs)
