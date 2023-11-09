# supernet.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 Advanced Micro Devices, Inc.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Use MS MXFP

TAU = 2.0

class Node(nn.Module):
    def __init__(self, nb_edges, propagates):
        super(Node, self).__init__()
        self.temp = TAU
        self.edges = nn.Parameter(torch.Tensor(nb_edges))
        self.propagates = propagates

    def combine(self, channels):
        return torch.sum(channels * self.p(), dim=-1)

    def p(self):
        return F.gumbel_softmax(self.edges, tau=self.temp, dim=0)

class Quant(Node):
    def __init__(self, dtypes):
        super(Quant, self).__init__(len(nb_edges=len(dtypes)), propagates=False)
        self.dtypes = dtypes

    def forward(self, x):
        channels = []
        for dtype in self.dtypes:
            new_tensor = # Use MS MXFP.quantize(x, dtype=dtype, axis=-1)
            channels.append(new_tensor)
        return self.combine(torch.stack(channels, dim=-1))

class Sparse(Node):
    def __init__(self, dense, blocksize):
        super(Sparse, self).__init__(nb_edges=len(dense), propagates=True)
        self.dense = dense
        self.blocksize = blocksize

    def forward(self, x):
        channels = []
        for dense in self.dense:
            if dense == self.blocksize:
                new_tensor = torch.ones_like(self.weight)
            else:
                new_tensor = # Use MS MXFP.sparsify(x, dense=dense, blocksize=self.blocksize, axis=-1)
            channels.append(new_tensor)
        return x * self.combine(torch.stack(channels, dim=-1))