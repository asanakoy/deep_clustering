import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SobelFilter(nn.Module):
    """
    Apply Sobel filter to  an input tensor
    Not trainable.


    Shape:
        - Input: :math:`(N, 3, H_{in}, W_{in})`
        - Output: :math:`(N, 2, H_{out}, W_{out})`
            where 0-th output channel is image gradient along Ox,
                  1-th output channel is image gradient along Oy.

    """

    def __init__(self):
        super(SobelFilter, self).__init__()

        gx = torch.Tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]).expand(3, 3, 3) / 3.0
        gy = torch.Tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]]).expand(3, 3, 3) / 3.0

        # nout(2) x nin(3) x 3 x 3
        self.weight = torch.nn.Parameter(torch.stack((gx, gy)))
        self.weight.requires_grad = False

    def forward(self, x):
        return F.conv2d(x, self.weight, bias=None, stride=1, padding=1)


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
