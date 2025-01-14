import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    def __init__(self, node_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.node_features = node_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(node_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features + 1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, node, edge, edge_feature):
        dv = "cuda:0" if node.is_cuda else "cpu"
        # dv = 'cpu'
        N = node.size()[0]
        edge = edge.t()
        assert not torch.isnan(edge).any()
        # print(input)

        h = torch.mm(node, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        # print(torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1))
        # print(edge_feature)
        edge_h = torch.cat(
            (torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1), edge_feature),
            dim=1,
        ).t()
        assert not torch.isnan(edge_h).any()
        # edge: (2*D + 1) x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # attention, edge_e: E

        e_rowsum = self.special_spmm(
            edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv)
        )
        # e_rowsum: N x 1

        # edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        #
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        h_prime = torch.where(
            torch.isnan(h_prime), torch.full_like(h_prime, 0), h_prime
        )
        h_prime = torch.add(h, h_prime)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        # print(h.size(), h_prime.size())

        if self.concat:
            # if this layer is not last layer,
            return [F.elu(h_prime), edge_e.reshape(edge_e.size()[0], 1)]
        else:
            # if this layer is last layer,
            return F.elu(h_prime)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.node_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


import torch
import torch.nn as nn
import torch.nn.functional as F

# from EGAT_layers import SpGraphAttentionLayer


# TODO: add prediction head
# TODO: experiment only predict binary and continuous values


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """
        Function Description:
        Initializes the model by defining the size of the feature space, and sets up layers for encoding decision variables, edge features, and constraint features.
        It includes two semi-convolutional attention layers and a final output layer.
        - nfeat: Initial feature dimension.
        - nhid: Dimension of the hidden layers.
        - nclass: Number of classes; for 0-1 integer programming, this would be 2.
        - dropout: Dropout rate.
        - alpha: Coefficient for leakyReLU.
        - nheads: Number of heads in the multi-head attention mechanism.
        Hint: Use the pre-written SpGraphAttentionLayer for the attention layers.
        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        embed_size = 64
        self.input_module = torch.nn.Sequential(
            torch.nn.Linear(nfeat, embed_size),
            # torch.nn.LogSoftmax(dim = 0),
        )
        self.attentions_u_to_v = [
            SpGraphAttentionLayer(
                embed_size, nhid, dropout=dropout, alpha=alpha, concat=True
            )
            for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions_u_to_v):
            self.add_module("attention_u_to_v_{}".format(i), attention)
        self.attentions_v_to_u = [
            SpGraphAttentionLayer(
                embed_size, nhid, dropout=dropout, alpha=alpha, concat=True
            )
            for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions_v_to_u):
            self.add_module("attention_v_to_u_{}".format(i), attention)

        self.out_att_u_to_v = SpGraphAttentionLayer(
            nhid * nheads, embed_size, dropout=dropout, alpha=alpha, concat=False
        )
        self.out_att_v_to_u = SpGraphAttentionLayer(
            nhid * nheads, embed_size, dropout=dropout, alpha=alpha, concat=False
        )
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(embed_size, embed_size),
            # torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_size, embed_size),
            # torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_size, nclass, bias=False),
            # torch.nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, edgeA, edgeB, edge_feat):
        """
        Function Description:
        Executes the forward pass using the provided constraint, edge, and variable features, processing them through an EGAT to produce the output.

        Parameters:
        - x: Features of the variable and constraint nodes.
        - edgeA, edgeB: Information about the edges.
        - edge_feat: Features associated with the edges.

        Return: The result after the forward propagation.
        """
        # print(x)
        x = self.input_module(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        # print(x)
        new_edge = torch.cat(
            [att(x, edgeA, edge_feat)[1] for att in self.attentions_u_to_v], dim=1
        )
        x = torch.cat(
            [att(x, edgeA, edge_feat)[0] for att in self.attentions_u_to_v], dim=1
        )
        x = self.out_att_u_to_v(x, edgeA, edge_feat)
        new_edge = torch.mean(new_edge, dim=1).reshape(new_edge.size()[0], 1)
        # x = self.softmax(x)
        new_edge_ = torch.cat(
            [att(x, edgeB, new_edge)[1] for att in self.attentions_v_to_u], dim=1
        )
        x = torch.cat(
            [att(x, edgeB, new_edge)[0] for att in self.attentions_v_to_u], dim=1
        )
        x = self.out_att_v_to_u(x, edgeB, new_edge)
        new_edge_ = torch.mean(new_edge_, dim=1).reshape(new_edge_.size()[0], 1)

        x = self.output_module(x)
        x = self.softmax(x)

        return x, new_edge_


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
