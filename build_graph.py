import scipy.sparse as sp
import torch
import numpy as np
import math
import scipy.sparse as sparse
from torch.nn.parameter import Parameter

nodes = []
node_trans = []

for i in range(4):
    nodes.append(torch.rand(1, 3))
size = len(nodes)
edges = torch.zeros(size, size)
for node in nodes:
    # N = torch.transpose(node, 0, 1).shape[0]
    N = node.shape[0]

    stdv = 1. / math.sqrt(node.size(1))
    weight = torch.empty(N, N).uniform_(-stdv, stdv)
    node = torch.transpose(node, 0, 1)
    # print(weight)
    node_trans.append(torch.mm(node, weight))

for i in range(len(node_trans)):
    for j in range(len(node_trans)):
        edges[i, j] = (torch.mm(torch.transpose(node_trans[i], 0, 1), node_trans[j]))

# adj

edge_sqrt = torch.zeros(size, size)
adj = torch.zeros(size, size)
# adj[0, 0] = 1
for n in range(len(edges)):
    # print(edge[0])
    for m in range(len(edges)):
        edge_sqrt[n, m] = edges[n][m]*edges[n][m]




edge_sqrt_sum = edge_sqrt.sum(1)
for n in range(size):
    for m in range(size):
        adj[n, m] = edge_sqrt[n][m] / edge_sqrt_sum[n]


# print(sum(adj[0]))

adj = adj.numpy()
# upper_X = sparse.triu(adj)
# adj_T = upper_X + upper_X.T - sp.diags(adj.diagonal())
# print(adj.sum(1))
# print(adj_T)
# print(adj_T.sum(1))


# def normalize(mx):
#     rowsum = np.array(mx.sum(1))  # degree D
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     mx = r_mat_inv.dot(mx)
#     return mx
def normalize(mx):
    degree = np.array(mx.sum(1))  # 为每个结点计算度
    degree = sp.diags(np.power(degree, -1).flatten())
    # print(degree.shape)
    mx = degree.dot(mx)
    return mx
# adj = adj.numpy()
adj = normalize(adj + sp.eye(adj.shape[0]))
# adj_1 = normalize(adj_T + sp.eye(adj_T.shape[0]))
adj = torch.tensor(adj)
# print(adj.sum(1))
# print(adj)
# print(adj_1)
# print(adj_1.sum(1))




