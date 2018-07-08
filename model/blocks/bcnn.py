# coding=utf-8

import torch
import torch.nn as nn

class BCNNBlock(nn.Module):
    """ Implements a single BCNN Block as described in this paper: 

        http://www.aclweb.org/anthology/Q16-1019
    """

    def __init__(self, conv, pool):
        """ Initializes the BCNN Block.

            Args:
                conv: Convolution module
                    The convolution layer for the BCNN Block.
                pool: Average Pooling module
                    The average pooling layer for the BCNN Block.

            Returns:
                None
        """
        super().__init__()
        self.conv = conv
        self.pool = pool

    def forward(self, x):
        """ Computes the forward pass over the BCNN Block.

            Args:
                x: torch.FloatTensor of shape (batch_size, 1, height, seq_len)
                    The input to the BCNN Block.

            Returns:
                out: torch.FloatTensor of shape (batch_size, 1, height, seq_len)
                    This output is passed to the next Block in the Model.
        """
        c = self.conv(x)
        out = self.pool(c)
        return out