# coding=utf-8

import torch
import torch.nn as nn

from model.pooling.allap import AllAP

class BCNNBlock(nn.Module):
    """ Implements a single BCNN Block as described in this paper: 

        http://www.aclweb.org/anthology/Q16-1019
    """

    def __init__(self, conv, pool, dropout_rate=0.5):
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
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x1, x2):
        """ Computes the forward pass over the BCNN Block.abs
            
            Args:
                x1, x2: torch.FloatTensors of shape (batch_size, 1, max_length, input_size)
                    The inputs to the BCNN Block.

            Returns:
                w1, w2: torch.FloatTensors of shape (batch_size, 1, max_length, output_size)
                    The outputs of the w-ap Average Pooling layer. These are passed to
                    the next Block.
                a1, a2: torch.FloatTensors of shape (batch_size, output_size)
                    The outputs of the all-ap Average Pooling layer. These are optionally
                    passed to the output layer.
        """
        c1, c2 = self.conv(x1), self.conv(x2)
        w1, w2 = self.pool(c1), self.pool(c2)
        a1, a2 = self.ap(c1), self.ap(c2)

        # Dropout
        w1 = self.dropout(w1) if self.training else w1
        w2 = self.dropout(w2) if self.training else w2
        return w1, w2, a1, a2
