# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class Convolution(nn.Module):
    """ Implements the convolution layer as described in this paper:

        http://www.aclweb.org/anthology/Q16-1019
    """
    def __init__(self, in_channels, out_channels, width, height):
        """ Initializes the convolution layer.

            Args:
                in_channels: int
                    The number of input channels.
                out_channels: int
                    The number of output channels. This is equal to the number
                    of convolution filters.
                width: int
                    The width of the convolution filters.
                height: int
                    The height of the convolution filters.
            
            Returns:
                None
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (height, width), 
                                stride=1, padding=width - 1)

    def forward(self, x):
        """ Computes the forward pass over the convolution layer.

            Args:
                x: torch.Tensor of shape (batch_size, in_channels, height, seq_len)
                    The input to the convolution layer.

            Returns:
                out: torch.Tensor of shape (batch_size, out_channels, height, seq_len + width - 1)
                    The output of the convolution layer.
        """
        out = F.tanh(self.conv(x))
        return out
        