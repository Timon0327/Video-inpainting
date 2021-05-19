import torch
import torch.nn as nn
import torch.nn.functional as F
import build_graph
import scipy.sparse as sp
import numpy as np
from numpy import linalg as LA
import cfgs.config as config


def square_normalize(input):
    '''
    A = e(xi,xj)^2/sum(e(xi,xj)^2)
    calculate normalized squared matrix
    :param input: matrix, tensor, [N, N]
    :return:
    '''
    square = input * input
    sum = torch.sum(square, dim=1, keepdim=True)
    return square / sum


def inverse_square_root(mat):
    '''
    calculate the 1/2 power of the inverse of the matrix
    :param mat: input matrix, ndarray, [N, N]
    :return: mat ^ -1/2, ndarray, [N, N]
    '''

    # check if it is square matrix
    assert mat.shape[0] == mat.shape[1]

    values, vecs = LA.eig(mat)

    new_values = 1 / values ** 0.5
    result = np.dot(np.dot(vecs, np.diag(new_values)), LA.inv(vecs))
    return result


class GCN(nn.Module):
    '''

    base class for STGCN

    '''
    def __init__(self, node_num, layers):
        '''

        :param node_num: number of nodes, int
        :param layers: number of layers, int
        '''
        self.node_num = node_num
        self.layers = layers

        # fully-connected layer in building adjacency matrix
        self.fc1 = nn.Linear(in_features=2048, out_features=2048, bias=False)

        # layers in the back
        self.blocks = []
        for k in range(layers):
            one = nn.Sequential(
                nn.Linear(in_features=2048, out_features=2048, bias=False),  # [nodes, 2048]
                nn.LayerNorm(normalized_shape=2048),  # [nodes, 2048]
                nn.LeakyReLU(negative_slope=0.1)  # [nodes, 2048]
            )
            self.blocks.append(one)

        self.laplacian_adj = None

    def adjacency_laplacian(self, input):
        '''
        build adjacency according to input features
        :param input: a batch of features, tensor, [nodes, channel]
        :return:
        '''
        # transformation
        out = self.fc1(input)   # out: [nodes, 2048]


        # adjacent matrix
        out = out * torch.t(out)  # out: [nodes, nodes]
        adj = square_normalize(out)  # adj: [nodes, nodes]

        # self-loop adjacency matrix
        I = torch.from_numpy(np.identity(self.node_num))
        I.requires_grad_(False)
        self_adj = adj + I

        # D
        sums = torch.sum(self_adj, dim=1)
        eig_values = 1 / sums ** 0.5
        D_sqt = torch.diag(eig_values)  # D_sqt is D^(-1/2), [nodes, nodes]

        self.laplacian_adj = torch.matmul(torch.matmul(D_sqt, self_adj), D_sqt)

    def forward(self, input):
        '''

        :param input:
        :return:
        '''
        # update adjacency matrix
        self.adjacency_laplacian(input)

        out = input
        for j in range(self.layers):
            out = torch.matmul(self.laplacian_adj, out)  # [nodes, 2048]
            out = self.blocks[j](out)  # [nodes, 2048]















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
