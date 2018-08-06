import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RavelTensor(nn.Module):
    """
    Reshape tensor [batch_size, s1, s2, ..., sn] into [batch_size, s1 *s2 * ... * sn]
    """
    def __init__(self, num_elements=None):
        super(RavelTensor, self).__init__()
        self.num_elements = num_elements

    def forward(self, x):
        if self.num_elements is None:
            self.num_elements = torch.prod(torch.tensor(x.size(), dtype=torch.int)[1:])
        return x.view(x.size(0), self.num_elements)
