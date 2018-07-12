# coding=utf-8

import torch
import torch.nn as nn

class ABCNN2Block(nn.Module):
    """ Implements a single ABCNN-2 block as described in this paper: 

        http://www.aclweb.org/anthology/Q16-1019
    """
    
    def __init__(self):
        super.__init__()
        pass

    def forward(self, x1, x2):
        """ Computes the forward pass over the ABCNN-1 Block.

            Args:
                x1, x2: torch.FloatTensors of shape (batch_size, 1, max_length, input_size)
                    The inputs to the ABCNN-1 Block.

            Returns:
                w1, w2: torch.FloatTensors of shape (batch_size, 1, max_length, output_size)
                    The outputs of the all-ap Average Pooling layer. These are
                    passed to the next Block.
                a1, a2: torch.FloatTensors of shape (batch_size, output_size)
                    The outputs of the all-ap Average Pooling layer. These are
                    optionally passed to the output layer.
        """
        pass