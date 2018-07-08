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

    def forward(self, in1, in2):
        """ Computes the forward pass over the ABCNN-2 Block.

            Args:
                in1, in2: torch.Tensor
                    The inputs to the ABCNN-2 Block.

            Returns:
                out1, out2: torch.Tensor
                    The outputs of the ABCNN-2 Block.
        """
        pass