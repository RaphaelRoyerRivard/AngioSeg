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


def get_all_node_pairs(graphs_path, point_pairs_path, view_pairs):
    all_node_pairs = {}
    # Load hand made node pairs
    image_pairs = load_node_pairs_from_path(point_pairs_path)
    # Loop over all view pairs
    for view_pair in view_pairs:
        print(view_pair)
        graph_data = []
        for view in view_pair:
            graph_file_path = graphs_path + f"{view}/{view}_graph_oriented.npy"
            graph_data.append(get_graph_data_from_numpy_graph_file(graph_file_path))

        # Convert point pairs to tensors so we can efficiently compute distances between image points and graph nodes
        point_pairs = torch.tensor(image_pairs[(view_pair[0], view_pair[1])], dtype=torch.float)

        # Associate pairs with nodes
        node_pairs = get_node_pairs(graph_data[0], graph_data[1], point_pairs)
        all_node_pairs[view_pair] = node_pairs

    return all_node_pairs


if __name__ == '__main__':
    graphs_path = f"C:/Users/Raphael/Pictures/skeleton/"
    point_pairs_path = r".\src\PAIRINGTOOL\pairs"
    view_pairs = [("1933060_LCA_-30_-25_2", "1933060_LCA_90_0_3"),
                  ("2022653_LCA_-30_-25_2", "2022653_LCA_90_0"),
                  ("3013714_LCA_-10_40", "3013714_LCA_90_0_2")]

    all_node_pairs = get_all_node_pairs(graphs_path, point_pairs_path, view_pairs)
    print(all_node_pairs)
