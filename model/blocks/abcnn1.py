# coding=utf-8

import torch
import torch.nn as nn

class ABCNN1Block(nn.Module):
    """ Implements a single ABCNN-1 Block as described in this paper: 

        http://www.aclweb.org/anthology/Q16-1019
    """
    
    def __init__(self, conv, attn, pool):
        """ Initializes the ABCNN-1 Block. 

            Args:
                conv: torch.nn.Conv1D
                    A 1D convolutional layer.
                attn: torch.nn.Module
                    An attention layer.
                pool: torch.nn.AvgPool1D
                    A 1D average pooling layer.
        """
        super().__init__()
        self.conv = conv
        self.attn = attn
        self.pool = pool

    def forward(self, x1, x2):
        """ Computes the forward pass over the ABCNN-1 Block.

            Args:
                x1, x2: torch.Tensor
                    The inputs to the ABCNN-1 Block.

            Returns:
                out1, out2: torch.Tensor
                    The outputs of the ABCNN-1 Block.
        """
        pass