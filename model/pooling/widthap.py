# coding=utf-8

import torch
import torch.nn as nn

class WidthAP(nn.Module):
    """ Implements the Average Pooling layer over windows of w columns (w-ap) 
        of a feature map as described in this paper:
        
        http://www.aclweb.org/anthology/Q16-1019
    """

    def __init__(self, width):
        """ Initializes the w-ap layer.

            Args:
                width: int
                    The width of the window.

            Returns:
                None
        """
        super().__init__()
        self.wp = nn.AvgPool2d((1, width), stride=(1, 1))

    def forward(self, x):
        """ Implements the forward pass over the w-ap layer. 
        
            Args:
                x: torch.Tensor of shape (batch_size, 1, height, seq_len + width - 1)
                    The output of the convolution layer.

            Returns:
                out: torch.Tensor of shape (batch_size, 1, height, seq_len)
                    The output of the w-ap layer.
        """
        return self.wp(x)
