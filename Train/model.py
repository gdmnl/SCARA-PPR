import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias='none'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias == 'bn':
            self.bias = nn.BatchNorm1d(out_features)
        else:
            self.bias = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        output = self.bias(output)
        # Residual connection
        if self.in_features == self.out_features:
            output += input
        return output


class MLP(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, bias):
        super(MLP, self).__init__()
        self.fcs = nn.ModuleList()
        self.fcs.append(Dense(nfeat, nhidden, bias))
        for _ in range(nlayers-2):
            self.fcs.append(Dense(nhidden, nhidden, bias))
        self.fcs.append(Dense(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act_fn(self.fcs[0](x))
        for fc in self.fcs[1:-1]:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn(fc(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return x


class MLP_sc(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, bias):
        super(MLP_sc, self).__init__()
        # Shortcut connection of input features
        self.shortcut = nn.Linear(nfeat, nhidden, bias=False)
        self.fcs = nn.ModuleList()
        self.fcs.append(Dense(nfeat, nhidden, bias))
        for _ in range(nlayers-2):
            self.fcs.append(Dense(nhidden, nhidden, bias))
        self.fcs.append(Dense(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def forward(self, x):
        out = F.dropout(x, self.dropout, training=self.training)
        out = self.fcs[0](out)
        out += self.shortcut(x)
        out = self.act_fn(out)
        for fc in self.fcs[1:-1]:
            out = F.dropout(out, self.dropout, training=self.training)
            out = fc(out)
            out += self.shortcut(x)
            out = self.act_fn(out)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.fcs[-1](out)
        return out
