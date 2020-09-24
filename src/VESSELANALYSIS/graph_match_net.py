import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.mlp = Sequential(Linear(2 * in_channels, out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)


class EdgeConvNet(nn.Module):
    def __init__(self, embedding_net):
        super(EdgeConvNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x):
        # (batch_size, channels, width, height)
        # print("MultiSiameseNet input", x.shape)
        embeddings = self.embedding_net(x)
        return embeddings


if __name__ == '__main__':
    edge_conv = EdgeConv(10, 10)
