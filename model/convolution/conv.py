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
                                stride=1, padding=(height - 1, 0))

    def forward(self, x):
        """ Computes the forward pass over the convolution layer.

            Args:
                x: torch.Tensor of shape (batch_size, 1, seq_len, height)
                    The input to the convolution layer.

            Returns:
                out: torch.Tensor of shape (batch_size, 1, width, out_channels)
                    The output of the convolution layer.
        """
        out = F.tanh(self.conv(x)) # shape (batch_size, out_channels, width, 1) 
        out = out.permute(0, 3, 2, 1) # shape (batch_size, 1, width, out_channels)
        return out
        
        # (64, 1, 20, 300)      convolution: in_channels = 1, out_channels = 50, height = 3, width = 300, stride = (1, 1), padding = (height - 1, 0)
        # ==> (64, 50, 22, 1)   tranpose: (0, 3, 2, 1)
        # ==> (64, 1, 22, 50)
