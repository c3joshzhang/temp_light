import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: add prediction head for handling continuous and binary
# TODO: experiment only predict binary and continuous values


class FocalLoss(nn.Module):

    def __init__(self, weight=None, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.weight = weight if weight is not None else torch.as_tensor([1, 1])

    def forward(self, preds, labels):
        preds = F.softmax(preds, dim=1)
        eps = 1e-7
        target = self.one_hot(preds.size(1), labels)
        ce = -1 * torch.log(preds + eps) * target
        floss = torch.pow((1 - preds), self.gamma) * ce
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)

    def one_hot(self, num, labels):
        one = torch.zeros((labels.size(0), num))
        one[range(labels.size(0)), labels] = 1
        return one


class Model(nn.Module):
    def __init__(self, n_node_feats, n_edge_feats, hidden_size):
        super().__init__()
        self.conv1 = dglnn.EdgeGATConv(n_node_feats, n_edge_feats, hidden_size, 8)
        self.conv2 = dglnn.EdgeGATConv(hidden_size * 8, n_edge_feats, hidden_size, 1)
        self.conv3 = dglnn.SAGEConv(hidden_size, 2, "mean")
        self.conv4 = dglnn.SAGEConv(hidden_size, 3, "mean")

    def forward(self, graph, node_x, edge_x):
        # TODO: partial graph conv
        node_x = self.conv1(graph, node_x, edge_x)
        node_x = F.relu(node_x)
        node_x = self.conv2(graph, node_x.reshape(node_x.size(0), -1), edge_x)
        node_x = node_x.reshape(node_x.size(0), -1)
        node_x = F.relu(node_x)

        prob = F.softmax(
            self.conv3(graph, node_x)
        )  # for completion to a high quality feasible solution
        dist = F.softmax(
            self.conv4(graph, node_x)
        )  # for distance to the optimal solution [-1, 0, 1] from 1-0, stay, 0-1
        return prob, dist
