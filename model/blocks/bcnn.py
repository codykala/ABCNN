# coding=utf-8

import torch
import torch.nn as nn

from model.pooling.allap import AllAP

class BCNNBlock(nn.Module):
    """ Implements a single BCNN Block as described in this paper: 

        http://www.aclweb.org/anthology/Q16-1019
    """

    def __init__(self, conv, pool):
        """ Initializes the BCNN Block.

            Args:
                conv: Convolution module
                    The Convolution layer for the BCNN Block.
                pool: Average Pooling module
                    The Average Pooling layer for the BCNN Block.

            Returns:
                None
        """
        super().__init__()
        self.conv = conv
        self.pool = pool
        self.ap = AllAP()

    def forward(self, x):
        """ Computes the forward pass over the BCNN Block.

            Args:
                x: torch.FloatTensor of shape (batch_size, 1, max_length, height)
                    The input to the BCNN Block.

            Returns:
                w_out: torch.FloatTensor of shape (batch_size, 1, max_length, out_channels)
                    This output is passed to the next Block in the Model.
                a_out: torch.FloatTensor of shape (batch_size, height)
                    This output is used to form the final representation returned
                    by the model.
        """
        c = self.conv(x) # shape (batch_size, 1, max_length + width - 1, out_channels)
        w_out = self.pool(c) # shape (batch_size, 1, max_length, out_channel)
        a_out = self.ap(c) # shape (batch_size, out_channels)
        return w_out, a_out