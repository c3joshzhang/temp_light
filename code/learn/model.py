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
        """
        preds: logits output values
        labels: labels
        """
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
    def __init__(self, n_node_feats, n_edge_feats, hidden_size, num_classes):
        super().__init__()
        self.conv1 = dglnn.EdgeGATConv(n_node_feats, n_edge_feats, hidden_size, 8)
        self.conv2 = dglnn.EdgeGATConv(hidden_size * 8, n_edge_feats, hidden_size, 1)
        self.conv3 = dglnn.SAGEConv(hidden_size, num_classes, "mean")

    def forward(self, graph, node_x, edge_x):
        node_x = self.conv1(graph, node_x, edge_x)
        node_x = F.relu(node_x)
        node_x = self.conv2(graph, node_x.reshape(node_x.size(0), -1), edge_x)
        node_x = node_x.reshape(node_x.size(0), -1)
        node_x = F.relu(node_x)
        node_x = self.conv3(graph, node_x)
        node_x = F.softmax(node_x)
        return node_x
