import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import DataLoader as GeometricDataLoader
from torch.optim import lr_scheduler
import numpy as np
from vesselanalysis import get_graph_data_from_raw_image_and_segmentation
from node_pairs_loader import load_node_pairs_from_path
from trainer import fit
from graph_match_net import GCN, SiameseLoss, SiameseAllPairsLoss


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


def get_all_graph_data(graphs_path, views, type='list'):
    all_graph_data = {} if type == 'dict' else []
    for view in views:
        graph_file_path = graphs_path + f"{view}/{view}_graph_oriented.npy"
        graph_data = get_graph_data_from_numpy_graph_file(graph_file_path)
        graph_data.view = view
        if type == 'dict':
            all_graph_data[view] = graph_data
        else:
            all_graph_data.append(graph_data)
    return all_graph_data


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


def get_all_positive_node_pairs(graphs_path, point_pairs_path, view_pairs):
    all_positive_node_pairs = {}
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
        nodes = image_pairs[(view_pair[0], view_pair[1])]
        point_pairs = torch.tensor(nodes, dtype=torch.float)

        # Associate pairs with nodes
        node_pairs = get_node_pairs(graph_data[0], graph_data[1], point_pairs)
        all_positive_node_pairs[view_pair] = node_pairs

    return all_positive_node_pairs


def get_all_negative_node_pairs(positive_pairs):
    all_negative_node_pairs = {}
    for view_pair, positive_node_pairs in positive_pairs.items():
        nodes_a = positive_node_pairs[:, 0]
        nodes_b = positive_node_pairs[:, 1]
        negative_node_pairs = np.array(np.meshgrid(nodes_a, nodes_b)).T.reshape(-1, 2)
        negative_node_pairs = np.delete(negative_node_pairs, positive_node_pairs, 0)
        all_negative_node_pairs[view_pair] = negative_node_pairs
    return all_negative_node_pairs


class NodePairsDataset(Dataset):
    def __init__(self, graphs_path, point_pairs_path, view_pairs, graphs_data, epoch_size, batch_size, random_seed=1234):
        self.view_pairs = view_pairs
        self.graphs_data = graphs_data
        self.epoch_size = epoch_size
        assert(batch_size % 2 == 0)
        self.batch_size = batch_size
        np.random.seed(random_seed)

        self.positive_node_pairs = get_all_positive_node_pairs(graphs_path, point_pairs_path, view_pairs)
        self.negative_node_pairs = get_all_negative_node_pairs(self.positive_node_pairs)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, item):
        return next(self.__iter__())

    def __iter__(self):
        epoch_count = 0
        while epoch_count < self.epoch_size:
            epoch_count += 1
            # Select a random view pair
            view_pair_index = np.random.randint(0, len(self.view_pairs))
            view_pair = self.view_pairs[view_pair_index]
            # Sample half the batch size of positive pairs and the other half with negative pairs
            potential_positive_pairs = self.positive_node_pairs[view_pair]
            potential_negative_pairs = self.negative_node_pairs[view_pair]
            sampling_count = int(min(len(potential_positive_pairs), self.batch_size / 2))
            sampled_positive_pairs_indices = np.random.choice(np.arange(len(potential_positive_pairs)), sampling_count, replace=False)
            sampled_negative_pairs_indices = np.random.choice(np.arange(len(potential_negative_pairs)), sampling_count, replace=False)
            sampled_positive_pairs = potential_positive_pairs[sampled_positive_pairs_indices]
            sampled_negative_pairs = potential_negative_pairs[sampled_negative_pairs_indices]
            # Yield the data for training
            yield view_pair, sampled_positive_pairs, sampled_negative_pairs  #, self.graphs_data[view_pair[0]], self.graphs_data[view_pair[1]]


if __name__ == '__main__':
    old_training = False
    graphs_path = f"C:/Users/Raphael/Pictures/skeleton/"
    point_pairs_path = r"..\PAIRINGTOOL\pairs"
    view_pairs = [("1933060_LCA_-30_-25_2", "1933060_LCA_90_0_3"),
                  ("2022653_LCA_-30_-25_2", "2022653_LCA_90_0"),
                  ("3013714_LCA_-10_40", "3013714_LCA_90_0_2")]

    # Load graphs data
    views = []
    for view_pair in view_pairs:
        if view_pair[0] not in views:
            views.append(view_pair[0])
        if view_pair[1] not in views:
            views.append(view_pair[1])
    # if old_training:
    #     data = get_all_graph_data(graphs_path, views, type='dict')
    # else:
    #     data = get_all_graph_data(graphs_path, views, type='list')
    data = get_all_graph_data(graphs_path, views, type='dict')

    # Create training and validation sets for node pairs
    if old_training:
        training_dataset = NodePairsDataset(graphs_path, point_pairs_path, view_pairs[:-1], data, epoch_size=10, batch_size=32)
        validation_dataset = NodePairsDataset(graphs_path, point_pairs_path, view_pairs[-1:], data, epoch_size=10, batch_size=32)
        train_loader = DataLoader(training_dataset, batch_size=1, shuffle=False, num_workers=0)
        val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        # train_loader = GeometricDataLoader(data[:-1], batch_size=1, shuffle=True)  # This doesn't work because we need batches to be pairs of graphs
        # val_loader = GeometricDataLoader(data[-1:], batch_size=1, shuffle=True)
        train_loader = DataLoader(view_pairs[:-1], batch_size=1, shuffle=True)
        val_loader = DataLoader(view_pairs[-1:], batch_size=1, shuffle=True)
        all_positive_node_pairs = get_all_positive_node_pairs(graphs_path, point_pairs_path, view_pairs)
        train_loader.all_positive_node_pairs = all_positive_node_pairs
        val_loader.all_positive_node_pairs = all_positive_node_pairs
        train_loader.graphs_data = data
        val_loader.graphs_data = data

    # Define other components needed for training
    # if old_training:
    #     input_size = next(iter(data.values())).x.shape[1]
    # else:
    #     input_size = data[0].x.shape[1]
    input_size = next(iter(data.values())).x.shape[1]
    model = GCN(input_size, 2)
    loss_fn = SiameseLoss() if old_training else SiameseAllPairsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, 100, gamma=0.9, last_epoch=-1)  # a gamma of 1 does nothing

    fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs=3000, log_interval=5, start_epoch=0, save_progress_path='./training_results', show_plots=False)
