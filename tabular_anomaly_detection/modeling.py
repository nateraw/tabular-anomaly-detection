from argparse import Namespace
from typing import List, Tuple, Union

import torch.nn.functional as F
from huggingface_hub import ModelHubMixin
from torch import nn


class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, activation=nn.LeakyReLU, **kwargs):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        self.activation = activation(**kwargs) if activation is not None else None

    def forward(self, x):
        if self.activation is None:
            return self.fc(x)
        return self.activation(self.fc(x))


class Encoder(nn.Module):
    def __init__(self, input_dim, *dims):
        super().__init__()
        dims = (input_dim,) + dims
        self.layers = nn.Sequential(
            *[Dense(dims[i], dims[i + 1], negative_slope=0.4, inplace=True) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, output_dim, *dims):
        super().__init__()
        self.layers = nn.Sequential(
            *[Dense(dims[i], dims[i + 1], negative_slope=0.4, inplace=True) for i in range(len(dims) - 1)]
            + [Dense(dims[-1], output_dim, activation=None)]
        )

    def forward(self, x):
        return self.layers(x)
