import torch
import torch.nn as nn
from torch.nn import init
# import build_graph
# import scipy.sparse as sp
import numpy as np
from numpy import linalg as LA
from cfgs import config


# def square_normalize(input):
#     '''
#     A = e(xi,xj)^2/sum(e(xi,xj)^2)
#     calculate normalized squared matrix
#     :param input: matrix, tensor, [N, nodes, nodes]
#     :return:
#     '''
#     square = input * input      # [N, nodes, nodes]
#     sum = torch.sum(square, dim=1, keepdim=True)    # [N, 1, nodes]
#     return square / sum     # [N, nodes, nodes]


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


class GCN_3(nn.Module):
    '''

    base class for STGCN

    '''
    def __init__(self, frames, slice, batch):
        '''
        layer 3
        node_num: number of nodes, int (usually 2N * SLICE * SLICE)
        :param frames: get 2 * 'frames' frames in total. int
        :param slice: how many patches each frame is divided into in each direction, int
        :param batch: batch size divided by gpu number, int
        '''
        super(GCN_3, self).__init__()
        self.node_num = 2 * frames * slice * slice
        self.frames = frames
        self.batch = batch
        self.slice = slice

        # fully-connected layer in building adjacency matrix
        self.fc1 = nn.Linear(in_features=2048, out_features=2048, bias=False)

        # layers in the back

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=2048, bias=False),  # [N, nodes, 2048]
            nn.LayerNorm(normalized_shape=2048),  # [N, nodes, 2048]
            nn.LeakyReLU(negative_slope=0.1)  # [N, nodes, 2048]
        )

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=2048, bias=False),  # [N, nodes, 2048]
            nn.LayerNorm(normalized_shape=2048),  # [N, nodes, 2048]
            nn.LeakyReLU(negative_slope=0.1)  # [N, nodes, 2048]
        )

        self.layer3 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=2048, bias=False),  # [N, nodes, 2048]
            nn.LayerNorm(normalized_shape=2048),  # [N, nodes, 2048]
            nn.LeakyReLU(negative_slope=0.1)  # [N, nodes, 2048]
        )

        self.conv1x1 = nn.Conv2d(in_channels=2 * self.frames, out_channels=1, kernel_size=1, stride=1)
        # for the case where slice is 2 and height of img is 640
        self.upsample = nn.Upsample(scale_factor=2., mode='bilinear')
        self.deconv = nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1, stride=3)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.laplacian_adj = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def adjacency_laplacian(self, input):
        '''
        build adjacency according to input features
        :param input: a batch of features, tensor, [N, nodes, channel]
        :return:
        '''
        # transformation
        out = self.fc1(input)   # out: [N, nodes, 2048]

        # adjacent matrix
        out = torch.matmul(out, torch.transpose(out, dim0=1, dim1=2))       # [N, nodes, nodes]
        adj = self.square_normalize(out)  # adj: [N, nodes, nodes]

        # self-loop adjacency matrix
        I = torch.from_numpy(np.identity(self.node_num)).to('cuda')    # [nodes, nodes]
        I.requires_grad_(False)
        self_adj = adj[:] + I       # [N, nodes, nodes]

        # D
        sums = torch.sum(self_adj, dim=1)   # [N, nodes]
        eig_values = 1 / sums ** 0.5
        D_sqt = torch.diag_embed(eig_values)  # D_sqt is D^(-1/2), [N, nodes, nodes]

        self.laplacian_adj = torch.matmul(torch.matmul(D_sqt, self_adj), D_sqt).float()     # [N, nodes, nodes]

    def square_normalize(self, input):
        '''
        A = e(xi,xj)^2/sum(e(xi,xj)^2)
        calculate normalized squared matrix
        :param input: matrix, tensor, [N, nodes, nodes]
        :return:
        '''
        square = input * input  # [N, nodes, nodes]
        sum = torch.sum(square, dim=1, keepdim=True)  # [N, 1, nodes]
        return square / sum  # [N, nodes, nodes]

    def forward(self, input):
        '''

        :param input: features, tensor, [N, nodes, channel]
        :return:
        '''
        # update adjacency matrix
        # print('input to gcn is', input.size())
        self.adjacency_laplacian(input)

        out = input     # [N, nodes, 2048]

        out = torch.matmul(self.laplacian_adj, out)  # [N, nodes, 2048]
        out = self.layer1(out)  # [N, nodes, 2048]

        out = torch.matmul(self.laplacian_adj, out)  # [N, nodes, 2048]
        out = self.layer2(out)  # [N, nodes, 2048]

        out = torch.matmul(self.laplacian_adj, out)  # [N, nodes, 2048]
        out = self.layer3(out)  # [N, nodes, 2048]

        out = torch.transpose(out, dim0=1, dim1=2)  # [N, 2048, nodes]
        # batch = out.size()[0]
        out = out.reshape(-1, 2 * self.frames, self.slice, self.slice)     # [N * 2048, 2 * frames, slice, slice]
        # print(out.size())

        out_conv1 = self.conv1x1(out)     # [N * 2048, 1, slice, slice]
        out = torch.squeeze(out_conv1)    # [N * 2048, slice, slice]

        out = out.view(self.batch, 2048, self.slice, self.slice)

        # for the case where slice is 2 and height of img is 640
        out = self.upsample(out)       # [N, 2048, 4, 4]
        out = self.deconv(out)         # [N, 1024, 10, 10]

        # out = torch.unsqueeze(out, dim=0)
        return out


# class GCN_spatial(nn.Module):
#     '''
#
#     Z = AXW
#
#     '''
#
#     def __init__(self, adj, feature_dim, out_dim):
#         super(GCN_spatial, self).__init__()
#         self.A = adj
#         # self.X = X
#         self.fc1 = nn.Linear(feature_dim, feature_dim, bias=False)
#         self.fc2 = nn.Linear(feature_dim, feature_dim, bias=False)
#         self.fc3 = nn.Linear(feature_dim, out_dim, bias=False)
#
#     def forward(self, X):
#         X = F.leaky_relu(self.fc1(adj.mm(X)))
#         X = F.leaky_relu(self.fc2(adj.mm(X)))
#         X = self.fc3(adj.mm(X))
#         maxpooling = torch.nn.MaxPool2d(2, stride=2)
#         X = maxpooling(X)
#         # print(X)
#         return X
#
# class spatial(nn.Module):
#     '''
#
#     Z = AXW
#
#     '''
#
#     def __init__(self, adj, feature_dim, out_dim):
#         super(GCN_spatial, self).__init__()
#         self.A = adj
#         # self.X = X
#         self.fc1 = nn.Linear(feature_dim, feature_dim, bias=False)
#         self.fc2 = nn.Linear(feature_dim, feature_dim, bias=False)
#         self.fc3 = nn.Linear(feature_dim, out_dim, bias=False)
#
#     def forward(self, X):
#         X = F.leaky_relu(self.fc1(adj.mm(X)))
#         X = F.leaky_relu(self.fc2(adj.mm(X)))
#         X = self.fc3(adj.mm(X))
#         # print('x', X)
#         maxpooling = torch.nn.MaxPool2d(2, stride=2)
#         output = maxpooling(X)
#         # print(X)
#         return output

'''
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
'''

if __name__ == '__main__':
    a = GCN_3(frames=config.N,
              slice=config.SLICE,
              batch=1)
    print('GCN created')





