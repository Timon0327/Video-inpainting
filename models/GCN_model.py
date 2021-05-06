import torch
import torch.nn as nn
import torch.nn.functional as F
import build_graph
import scipy.sparse as sp
import numpy as np


class GCN_spatial(nn.Module):
    '''

    Z = AXW

    '''

    def __init__(self, adj, feature_dim, out_dim):
        super(GCN_spatial, self).__init__()
        self.A = adj
        # self.X = X
        self.fc1 = nn.Linear(feature_dim, feature_dim, bias=False)
        self.fc2 = nn.Linear(feature_dim, feature_dim, bias=False)
        self.fc3 = nn.Linear(feature_dim, out_dim, bias=False)

    def forward(self, X):
        X = F.leaky_relu(self.fc1(adj.mm(X)))
        X = F.leaky_relu(self.fc2(adj.mm(X)))
        X = self.fc3(adj.mm(X))
        maxpooling = torch.nn.MaxPool2d(2, stride=2)
        X = maxpooling(X)
        # print(X)
        return X

class spatial(nn.Module):
    '''

    Z = AXW

    '''

    def __init__(self, adj, feature_dim, out_dim):
        super(GCN_spatial, self).__init__()
        self.A = adj
        # self.X = X
        self.fc1 = nn.Linear(feature_dim, feature_dim, bias=False)
        self.fc2 = nn.Linear(feature_dim, feature_dim, bias=False)
        self.fc3 = nn.Linear(feature_dim, out_dim, bias=False)

    def forward(self, X):
        X = F.leaky_relu(self.fc1(adj.mm(X)))
        X = F.leaky_relu(self.fc2(adj.mm(X)))
        X = self.fc3(adj.mm(X))
        # print('x', X)
        maxpooling = torch.nn.MaxPool2d(2, stride=2)
        output = maxpooling(X)
        # print(X)
        return output


dataset = build_graph.nodes
new_data = torch.zeros(len(dataset), dataset[0].shape[1], dtype=torch.float32)
for i in range(len(dataset)):
    new_data[i] = dataset[i]

# print(new_data)

feature_dim = 3
# print(feature_dim)
out_dim = 2048
adj = build_graph.adj  # print(adj)
# print(adj)
model_temporal = GCN_temporal(adj_t, feature_dim, out_dim)
model_sptial = GCN_spatial(adj_s, feature_dim, out_dim)
# feature_spa = model_spa(new_data)
# a = adj.mm(new_data)
