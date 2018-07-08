# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class AllAP(nn.Module):
    """ Implements the Average Pooling layer over all columns (all-ap) of a 
        feature map as described in this paper:
        
        http://www.aclweb.org/anthology/Q16-1019
    """

    def forward(self, x):
        """ Computes the forward pass over the all-ap layer.

            Args:
                x: torch.Tensor of shape (batch_size, 1, height, seq_len)
                    The output of the convolution layer.

            Returns:
                out: torch.Tensor of shape (batch_size, height)
                    The output of the all-ap layer.
        """
        seq_len = x.shape[3]
        out = F.avg_pool2d(x, (1, seq_len)) # shape (batch_size, 1, height, 1)
        out = torch.squeeze(out) # shape (batch_size, height)
        return out

