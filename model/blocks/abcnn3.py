# coding=utf-8

import torch
import torch.nn as nn

class ABCNN3Block(nn.Module):
    """ Implements a single ABCNN-3 block as described in this paper: 

        http://www.aclweb.org/anthology/Q16-1019
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, in1, in2):
        """ Computes the forward pass over the ABCNN-3 Block.

            Args:
                in1, in2: torch.Tensor
                    The inputs to the ABCNN-3 Block.

            Returns:
                out1, out2: torch.Tensor
                    The outputs of the ABCNN-3 Block.
        """
        pass