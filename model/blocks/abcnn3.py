# coding=utf-8

import torch
import torch.nn as nn

from model.pooling.allap import AllAP

class ABCNN3Block(nn.Module):
    """ Implements a single ABCNN-3 block as described in this paper: 

        http://www.aclweb.org/anthology/Q16-1019
    """
    
    def __init__(self, attn1, conv, attn2, dropout_rate=0.5):
        """ Initializes the ABCNN-3 Block.

            Args:
                attn1: ABCNN1Attention Module
                    The attention mechanism used by the ABCNN-1 Block.
                conv: Convolution Module
                    The convolutional layer.
                attn2: ABCNN2Attention Module
                    The attention mechanism used by the ABCNN-2 Block
                    (also called the attention-based average pooling layer).
                
            Returns:
                None
        """
        super().__init__()
        self.attn1 = attn1
        self.conv = conv
        self.attn2 = attn2
        self.ap = AllAP()
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x1, x2):
        """ Computes the forward pass over the ABCNN-3 Block.

            Args:
                x1, x2: torch.Tensors of shape (batch_size, 1, max_length, input_size)
                    The inputs to the ABCNN-3 Block.

            Returns:
                w1, w2: torch.Tensors of shape (batch_size, 1, max_length, output_size)
                    The outputs of the atention-based average pooling layer. These
                    are passed to the next Block.
                a1, a2: torch.Tensors of shape (batch_size, output_size)
                    The outputs of the all-ap average pooling layer. These are
                    optionally passed to the output layer.
        """
        o1, o2 = self.attn1(x1, x2)
        c1, c2 = self.conv(o1), self.conv(o2)
        w1, w2 = self.attn2(c1, c2)
        a1, a2 = self.ap(c1), self.ap(c2)

        # Dropout
        w1 = self.dropout(w1) if self.training else w1
        w2 = self.dropout(w2) if self.training else w2
        return w1, w2, a1, a2
