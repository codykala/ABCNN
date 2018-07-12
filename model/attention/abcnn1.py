# coding=utf-8

import torch
import torch.nn as nn

from model.attention.utils import compute_attention_matrix
from model.attention.utils import cosine
from model.attention.utils import euclidean
from model.attention.utils import manhattan

class ABCNN1Attention(nn.Module):
    """ Implements the attention mechanism for the ABCNN-1 model described 
        in this paper:

        http://www.aclweb.org/anthology/Q16-1019
    """

    def __init__(self, input_size, max_length, share_weights, match_score):
        """ Initializes the parameters of the attention layer for the ABCNN-1
            Block.
        
            Args:
                input_size: int
                    The dimension of the input features.
                max_length: int
                    The length of the sequences.
                share_weights: boolean
                    Specifies whether to share the weights.
                match_score: string
                    The name of the function used to compute the entries of the 
                    attention feature map. Should be "euclidean" or "cosine".
            
            Returns:
                None
        """
        super().__init__()
        
        # Initialize weights
        W = nn.Parameter(torch.Tensor(max_length, input_size))
        W1 = nn.Parameter(torch.Tensor(max_length, input_size))
        W2 = nn.Parameter(torch.Tensor(max_length, input_size))
        self.W1 = W if share_weights else W1
        self.W2 = W if share_weights else W2
        
        # Choose match-score function
        functions = {
            "cosine": cosine,
            "euclidean": euclidean,
            "manhattan": manhattan
        }
        self.match_score = functions[match_score]

    def forward(self, x1, x2):
        """ Computes the forward pass for the attention layer of the ABCNN-1
            Block.

            Args:
                x1, x2: torch.Tensors of shape (batch_size, 1, max_length, input_size)
                    The inputs to the ABCNN-1 Block.

            Returns:
                attn1, attn2: torch.Tensors of shape (batch_size, 2, max_length, input_size)
                    The output of the attention layer for the ABCNN-1 Block.
        """
        # Get attention matrix and its transpose
        A = compute_attention_matrix(x1, x2, self.match_score)
        A_t = A.permute(0, 1, 3, 2)

        # Compute attention feature maps
        a1 = torch.matmul(A, self.W1) 
        a2 = torch.matmul(A_t, self.W2)

        # Stack attention feature maps with inputs
        attn1 = torch.cat([x1, a1], dim=1)
        attn2 = torch.cat([x2, a2], dim=1)
        return attn1, attn2