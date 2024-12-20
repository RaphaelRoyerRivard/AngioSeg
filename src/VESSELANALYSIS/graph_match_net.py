import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import KarateClub, TUDataset
from torch_geometric.nn import EdgeConv, GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_networkx
import networkx as nx
from matplotlib import pyplot as plt
from os import walk
import sys
import time


class NodeEdgeConv(MessagePassing):
    """
    This class implements an edge convolution that uses both features of the connected nodes and features of the edges
    connecting the nodes.
    """
    def __init__(self, node_features_size, edge_features_size, out_channels):
        super(NodeEdgeConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.mlp = Sequential(Linear(2 * node_features_size + edge_features_size, out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, node_features_size] (N is the number of nodes and node_features_size is the number of node features)
        # edge_index has shape [2, E] (E is the number of edges)

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, e):
        # x_i has shape [E, node_features_size] (node features from which the message pass from)
        # x_j has shape [E, node_features_size] (node features from which the message go to)
        # e has shape [E, edge_features_size] (edges features from which the message go to)

        tmp = torch.cat([x_i, x_j - x_i, e], dim=1)  # tmp has shape [E, 2 * node_features_size + edge_features_size
        return self.mlp(tmp)


class SiameseEdgeConvNet(nn.Module):
    """
    Modified from https://github.com/shz9/graph-alignment/blob/master/model.py
    """
    def __init__(self, input_size, hidden_layer_size, output_size, normalize=False):
        super(SiameseEdgeConvNet, self).__init__()

        self.normalize_embed = normalize

        # TODO find how to use the NodeEdgeConv
        self.conv1 = EdgeConv(input_size, hidden_layer_size)
        self.layer1activation = nn.PReLU(hidden_layer_size)
        self.conv2 = EdgeConv(hidden_layer_size, output_size)

    def embed_graph(self, g):

        x, edge_index = g.x, g.edge_index

        x = self.layer1activation(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        if self.normalize_embed:
            return F.normalize(x, p=2, dim=-1)
        else:
            return x

    def forward(self, g1, g2):
        x1 = self.embed_graph(g1)
        x2 = self.embed_graph(g2)
        return x1, x2


class GCN(torch.nn.Module):
    """
    Modified from https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8?usp=sharing#scrollTo=AkQAVluLuxT_
    """
    def __init__(self, input_size, output_size):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_size, 4)
        self.a1 = nn.PReLU(4)
        self.conv2 = GCNConv(4, 4)
        self.a2 = nn.PReLU(4)
        self.conv3 = GCNConv(4, 2)
        self.a3 = nn.PReLU(2)
        self.lin = Linear(2, output_size)

    def forward(self, x, edge_index):
        h = self.a1(self.conv1(x, edge_index))
        h = self.a2(self.conv2(h, edge_index))
        h = self.a3(self.conv3(h, edge_index))

        out = self.lin(h)

        return out, h


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, x):
        embedding_size = x.shape[1]
        batch_size = x.shape[0]

        normalized_x = F.normalize(x, dim=-1)

        cosim = torch.bmm(normalized_x.view(1, batch_size, embedding_size),
                          normalized_x.t().view(1, embedding_size, batch_size)).cpu()
        cosim = torch.squeeze(cosim)
        cosim = (cosim + 1) / 2
        cosim -= torch.eye(batch_size)
        cosim = cosim ** 2

        return torch.mean(cosim)


class ClampedSquaredDistanceLoss(nn.Module):
    def __init__(self):
        super(ClampedSquaredDistanceLoss, self).__init__()

    def forward(self, x):
        normalized_x = F.normalize(x, dim=-1)
        dist = torch.cdist(normalized_x, normalized_x)
        dist = torch.clamp_max(dist, 0.25)
        return -torch.mean(dist)


class DoubleLoss(nn.Module):
    def __init__(self):
        super(DoubleLoss, self).__init__()

    def forward(self, x):
        embedding_size = x.shape[1]
        batch_size = x.shape[0]

        normalized_x = F.normalize(x, dim=-1)

        cosim = torch.bmm(normalized_x.view(1, batch_size, embedding_size),
                          normalized_x.t().view(1, embedding_size, batch_size)).cpu()
        cosim = torch.squeeze(cosim)
        cosim = (cosim + 1) / 2
        cosim -= torch.eye(batch_size)
        cosim = cosim ** 2

        dist = torch.cdist(normalized_x, normalized_x)

        print("cosim", torch.mean(cosim), "dist", -torch.mean(dist))
        return torch.mean(cosim) - torch.mean(dist)


class SiameseLoss(nn.Module):
    def __init__(self):
        super(SiameseLoss, self).__init__()

    def forward(self, x1, x2, gt):
        normalized_x1 = F.normalize(x1, dim=-1)
        normalized_x2 = F.normalize(x2, dim=-1)
        dist = torch.abs(normalized_x1 - normalized_x2)
        # dist = torch.abs(x1 - x2)
        dist = dist ** 2
        summed_dists = torch.sum(dist, dim=1)
        loss = summed_dists * gt
        return -torch.mean(loss)


class SiameseAllPairsLoss(nn.Module):
    def __init__(self):
        super(SiameseAllPairsLoss, self).__init__()

    def forward(self, x1, x2, positive_pairs):
        use_normalized_vectors = False
        if use_normalized_vectors:
            normalized_x1 = F.normalize(x1, dim=-1)
            normalized_x2 = F.normalize(x2, dim=-1)
            dist = torch.cdist(normalized_x1, normalized_x2)
        else:
            dist = torch.cdist(x1, x2)
        # we create the ground truth matrix that sets a value of 1 for every negative pair and a value inversely proportional to the quantity of positive pairs compared to the total number of pairs
        gt = torch.ones_like(dist)
        gt[positive_pairs[:, 0], positive_pairs[:, 1]] = -(dist.shape[0] * dist.shape[1] / positive_pairs.shape[0])
        loss = dist * gt
        if not use_normalized_vectors:
            loss = torch.clamp_max(loss, 2)
        return -torch.mean(loss)


class SiameseAbsolutelyAllPairsLoss(nn.Module):
    def __init__(self):
        super(SiameseAbsolutelyAllPairsLoss, self).__init__()

    def forward(self, x1, x2, positive_pairs):
        use_normalized_vectors = False
        if use_normalized_vectors:
            normalized_x1 = F.normalize(x1, dim=-1)
            normalized_x2 = F.normalize(x2, dim=-1)
            dist_1_2 = torch.cdist(normalized_x1, normalized_x2)
            dist_1 = torch.cdist(normalized_x1, normalized_x1)
            dist_2 = torch.cdist(normalized_x2, normalized_x2)
        else:
            dist_1_2 = torch.cdist(x1, x2)
            dist_1 = torch.cdist(x1, x1)
            dist_2 = torch.cdist(x2, x2)
        # we create the ground truth matrix that sets a value of 1 for every negative pair and a value inversely proportional to the quantity of positive pairs compared to the total number of pairs
        gt = torch.ones_like(dist_1_2)
        gt[positive_pairs[:, 0], positive_pairs[:, 1]] = -(dist_1_2.shape[0] * dist_1_2.shape[1] / positive_pairs.shape[0])
        loss_1_2 = dist_1_2 * gt
        loss_1 = dist_1
        loss_2 = dist_2
        if not use_normalized_vectors:
            max_clamp = 2
            loss_1_2 = torch.clamp_max(loss_1_2, max_clamp)
            loss_1 = torch.clamp_max(loss_1, max_clamp)
            loss_2 = torch.clamp_max(loss_2, max_clamp)
        print(int(1000 * torch.mean(loss_1_2))/1000, int(1000 * torch.mean(loss_1))/1000, int(1000 * torch.mean(loss_2))/1000)
        total_loss = -torch.mean(loss_1_2) + -torch.mean(loss_1)/2 + -torch.mean(loss_2)/2

        add_feature_vector_norm_regularization = False
        if add_feature_vector_norm_regularization:
            norm_1 = torch.norm(x1, dim=-1)
            norm_2 = torch.norm(x2, dim=-1)
            total_loss += torch.mean(norm_1) + torch.mean(norm_2)

        add_cosine_similarity_loss = False
        if add_cosine_similarity_loss:
            batch_size_1, embedding_size_1 = x1.shape
            batch_size_2, embedding_size_2 = x2.shape

            normalized_x1 = F.normalize(x1, dim=-1)
            normalized_x2 = F.normalize(x2, dim=-1)

            # calculate cosine similarity for every combination
            cosine_similarities_1 = torch.bmm(normalized_x1.view(1, batch_size_1, embedding_size_1),
                                              normalized_x1.t().view(1, embedding_size_1, batch_size_1))
            cosine_similarities_2 = torch.bmm(normalized_x2.view(1, batch_size_2, embedding_size_2),
                                              normalized_x2.t().view(1, embedding_size_2, batch_size_2))

            normalized_cosine_similarities_1 = (cosine_similarities_1 + 1) / 2
            normalized_cosine_similarities_2 = (cosine_similarities_2 + 1) / 2

            normalized_cosine_similarities_1 = torch.clamp_max(normalized_cosine_similarities_1, 0.01)
            normalized_cosine_similarities_2 = torch.clamp_max(normalized_cosine_similarities_2, 0.01)

            total_loss += 100 * (torch.mean(normalized_cosine_similarities_1) + torch.mean(normalized_cosine_similarities_2))

        return total_loss


def visualize(h, color='r', epoch=None, loss=None, title=""):
    """
    Copied from https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8?usp=sharing#scrollTo=AkQAVluLuxT_
    """
    plt.figure(figsize=(7,7))

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
        plt.grid(True, axis='both')
    else:
        plt.xticks([])
        plt.yticks([])
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False, node_color=color, cmap="Set2")
    plt.title(title)
    plt.show()


def train_on_data(model, data, loss_function, optimizer):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = loss_function(out)
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, out, h


def train_with_dataloader(model, train_loader, loss_function, optimizer):
    model.train()

    losses = []
    outs = None
    hs = None
    for data in train_loader:  # Iterate in batches over the training dataset.
        # out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = loss_function(out)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        losses.append(loss)
        outs = out if outs is None else torch.cat([outs, out])
        hs = out if hs is None else torch.cat([outs, out])

    losses = torch.tensor(losses)
    return losses.mean(), outs, hs


def test_with_dataloader(model, test_loader, loss_function):
    model.eval()

    outs = None
    losses = []
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        out, h = model(data.x, data.edge_index)
        loss = loss_function(out)  # Compute the loss.
        losses.append(loss)
        outs = out if outs is None else torch.cat([outs, out])

    losses = torch.tensor(losses)
    return losses.mean(), outs


def train_on_single_graph():
    # Load graph of 1933060 #1
    raw_image_path, segmented_image_path = get_image_paths(patient_angle="1933060_LCA_-30_-25_2")
    data = get_graph_data_from_images(raw_image_path, segmented_image_path, oriented=True)
    print(data)
    G = to_networkx(data, to_undirected=False)
    visualize(G)
    # visualize(data.x[:, :2])

    input_size = data.x.shape[1]
    model = GCN(input_size, 2)
    out, h = model(data.x, data.edge_index)
    print(f'Embedding shape: {list(h.shape)}')

    # visualize(h, color=data.y)

    # loss_function = SquaredDistanceLoss()
    # loss_function = CosineSimilarityLoss()
    loss_function = DoubleLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    loss = 0
    epoch = 0
    for epoch in range(1001):
        loss, out, h = train_on_data(model, data, loss_function, optimizer)
        losses.append(loss)
        print(epoch, loss)
        # if epoch % 100 == 0:
        #     visualize(h, color=data.y, epoch=epoch, loss=loss)
    plt.plot(losses)
    plt.title("Loss progression")
    plt.show()
    visualize(out, color=data.y, epoch=epoch, loss=loss, title="Final embeddings with normalized distance loss")
    normalized_out = F.normalize(out, dim=-1)
    visualize(normalized_out, color=data.y, epoch=epoch, loss=loss, title="Normalized final embeddings with normalized distance loss")

    # Test on another graph
    raw_image_path, segmented_image_path = get_image_paths(patient_angle="1933060_LCA_90_0_3")
    data = get_graph_data_from_images(raw_image_path, segmented_image_path, oriented=True)
    out, h = model(data.x, data.edge_index)
    loss = loss_function(out)
    print("loss on new data", loss)
    normalized_out = F.normalize(out, dim=-1)
    visualize(normalized_out, color=data.y, title="Normalized final embeddings of another graph")


def train_on_multiple_graphs():
    start_time = time.time()
    dataset = []
    for path, subfolders, files in walk(DATA_PATH):
        folder = path.split("\\")[-1].split("/")[-1]
        # if folder + ".tif" not in files:
        if folder + "_graph_oriented.npy" not in files:
            continue
        # Load graph
        # raw_image_path, segmented_image_path = get_image_paths(patient_angle=folder)
        # data = get_graph_data_from_images(raw_image_path, segmented_image_path, oriented=True)
        data = get_graph_data_from_numpy_graph_file(path + "/" + folder + "_graph_oriented.npy")
        dataset.append(data)

    print(f"getting graph data took {time.time() - start_time}s")
    print(len(dataset), "graphs in dataset")

    input_size = dataset[0].x.shape[1]
    model = GCN(input_size, 2)

    loss_function = ClampedSquaredDistanceLoss()
    # loss_function = CosineSimilarityLoss()
    # loss_function = DoubleLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # train_loader = DataLoader(dataset[:60], batch_size=5, shuffle=True)
    # val_loader = DataLoader(dataset[60:], batch_size=5, shuffle=True)
    train_loader = DataLoader(dataset[:3], batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset[-3:-1], batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset[-1:], batch_size=1, shuffle=True)
    losses = []
    val_losses = []
    min_val_loss = 1e10  # big number
    out = None
    val_out = None
    epoch = 0

    # Show output before training
    val_loss, val_out = test_with_dataloader(model, train_loader, loss_function)
    normalized_out = F.normalize(val_out, dim=-1)
    visualize(normalized_out, color='b', title="Normalized untrained embeddings")

    # Train
    for epoch in range(1, 201):
        loss, out, h = train_with_dataloader(model, train_loader, loss_function, optimizer)
        losses.append(loss)
        val_loss, val_out = test_with_dataloader(model, val_loader, loss_function)
        val_losses.append(val_loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), './training_state.pth')
        # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    plt.plot(losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.title("Loss progression")
    plt.legend()
    plt.show()

    model.load_state_dict(torch.load('./training_state.pth'))

    loss, out = test_with_dataloader(model, train_loader, loss_function)
    visualize(out, color='b', epoch=epoch, loss=losses[-1], title="Final embeddings")
    normalized_out = F.normalize(out, dim=-1)
    visualize(normalized_out, color='b', epoch=epoch, loss=losses[-1], title="Normalized final embeddings")

    val_loss, val_out = test_with_dataloader(model, val_loader, loss_function)
    normalized_val_out = F.normalize(val_out, dim=-1)
    visualize(normalized_val_out, color='b', title="Normalized embeddings of a graph from validation set")

    test_loss, test_out = test_with_dataloader(model, test_loader, loss_function)
    normalized_test_out = F.normalize(test_out, dim=-1)
    visualize(normalized_test_out, color='b', title="Normalized embeddings of a graph from test set")


def get_image_paths(whole_path=None, patient_angle=None):
    images_path = whole_path if whole_path is not None else f"{DATA_PATH}\\{patient_angle}\\"
    raw_image_path = images_path + f"{patient_angle}.tif"
    segmented_image_path = images_path + f"{patient_angle}_seg.tif"
    return raw_image_path, segmented_image_path


if __name__ == '__main__':
    DATA_PATH = r"C:\Users\Raphael\Pictures\skeleton"

    # KarateClub dataset
    # dataset = KarateClub()
    # data = dataset[0]  # Get the first graph object.
    # G = to_networkx(data, to_undirected=True)
    # visualize(G, color=data.y)

    # Batch loading with MUTAG dataset
    # dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    # train_dataset = dataset[:150]
    # test_dataset = dataset[150:]
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    #
    # for step, data in enumerate(train_loader):
    #     print(f'Step {step + 1}:')
    #     print('=======')
    #     print(f'Number of graphs in the current batch: {data.num_graphs}')
    #     print(data)
    #     print()

    from siamese_trainer import get_graph_data_from_images, get_graph_data_from_numpy_graph_file
    train_on_multiple_graphs()
