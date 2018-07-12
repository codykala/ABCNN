# coding=utf-8

import torch
import torch.nn as nn

from model.pooling.allap import AllAP

class ABCNN2Block(nn.Module):
    """ Implements a single ABCNN-2 block as described in this paper: 

        http://www.aclweb.org/anthology/Q16-1019
    """
    
    def __init__(self, conv, attn):
        super().__init__()
        self.conv = conv
        self.attn = attn
        self.ap = AllAP()

    def forward(self, x1, x2):
        """ Computes the forward pass over the ABCNN-1 Block.

            Args:
                x1, x2: torch.Tensors of shape (batch_size, 1, max_length, input_size)
                    The inputs to the ABCNN-1 Block.

            Returns:
                w1, w2: torch.Tensors of shape (batch_size, 1, max_length, output_size)
                    The outputs of the all-ap Average Pooling layer. These are
                    passed to the next Block.
                a1, a2: torch.Tensors of shape (batch_size, output_size)
                    The outputs of the all-ap Average Pooling layer. These are
                    optionally passed to the output layer.
        """
        c1, c2 = self.conv(x1), self.conv(x2)
        w1, w2 = self.attn(c1, c2)
        a1, a2 = self.ap(c1), self.ap(c2)
        return w1, w2, a1, a2
