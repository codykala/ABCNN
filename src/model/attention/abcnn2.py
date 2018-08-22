# coding=utf-8

import torch
import torch.nn as nn

from model.attention.utils import compute_attention_matrix
from model.attention.utils import cosine
from model.attention.utils import euclidean
from model.attention.utils import manhattan

# Use GPU if available, otherwise use CPU
USE_CUDA = torch.cuda.is_available()

class ABCNN2Attention(nn.Module):
    """ Implements the attention mechanism for the ABCNN-2 model described 
        in this paper:

        http://www.aclweb.org/anthology/Q16-1019
    """

    def __init__(self, max_length, width, match_score):
        """ Initializes the parameters of the attention layer for the ABCNN-2
            Block.

            Args:
        """
        super().__init__()
        self.max_length = max_length
        self.width = width
        functions = {
            "cosine": cosine,
            "euclidean": euclidean,
            "manhattan": manhattan
        }
        self.match_score = functions[match_score]

    def forward(self, x1, x2):
        """ Computes the forward pass for the attention layer of the ABCNN-2
            Block.

            Args:
                x1, x2: torch.Tensors of shape (batch_size, 1, max_length + width - 1, output_size)
                    The outputs from the convolutional layer.

            Returns:
                w1, w2: torch.Tensors of shape (batch_size, 1, max_length, output_size)
                    The outputs from the attention layer. This layer takes
                    the place of the Average Pooling layer seen in the BCNN and ABCNN-1
                    models.
        """
        # Compute attention matrix for outputs of convolutional layer
        A = compute_attention_matrix(x1, x2, self.match_score)

        # Initialize outputs for attention layer
        batch_size = x1.shape[0]
        output_size = x1.shape[3]
        w1 = torch.zeros((batch_size, 1, self.max_length, output_size))
        w2 = torch.zeros((batch_size, 1, self.max_length, output_size))
        w1 = w1.cuda() if USE_CUDA else w1
        w2 = w2.cuda() if USE_CUDA else w2

        # Compute the outputs
        for j in range(self.max_length):
            for k in range(j, j + self.width):    
                row_sum = torch.sum(A[:, :, :, k], dim=2, keepdim=True)
                col_sum = torch.sum(A[:, :, k, :], dim=2, keepdim=True)
                w1[:, :, j, :] += row_sum * x1[:, :, k, :]
                w2[:, :, j, :] += col_sum * x2[:, :, k, :]
        return w1, w2
