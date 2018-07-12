# coding=utf-8

import torch
import torch.nn as nn

# Use GPU if available, otherwise use CPU
USE_CUDA = torch.cuda.is_available()

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
        A = self._compute_attention_matrix(x1, x2)
        A_t = A.permute(0, 1, 3, 2)

        # Compute attention feature maps
        a1 = torch.matmul(A, self.W1) 
        a2 = torch.matmul(A_t, self.W2)

        # Stack attention feature maps with inputs
        attn1 = torch.cat([x1, a1], dim=1)
        attn2 = torch.cat([x2, a2], dim=1)
        return attn1, attn2 

    def _compute_attention_matrix(self, x1, x2):
        """ Computes the attention feature map for the batch of inputs x1 and x2.

            The following description is taken directly from the ABCNN paper:

            Let F_{i, r} in R^{d x s} be the representation feature map of
            sentence i (i in {0, 1}). Then we define the attention matrix A in R^{s x s}
            as follows:

                A_{i, j} = match-score(F_{0, r}[:, i], F_{1, r}[:, j])

            Args:
                x1, x2: torch.Tensors of shape (batch_size, 1, max_length, input_size)
                    A batch of input tensors.

            Returns:
                A: torch.Tensor of shape (batch_size, 1, max_length, max_length)
                    A batch of attention feature maps.
        """
        batch_size = x1.shape[0]
        max_length = x1.shape[2]
        A = torch.empty((batch_size, 1, max_length, max_length), dtype=torch.float)
        A = A.cuda() if USE_CUDA else A # move to GPU
        for i in range(max_length):
            for j in range(max_length):
                b1 = x1[:, :, i, :]
                b2 = x2[:, :, j, :]
                A[:, :, i, j] = self.match_score(b1, b2)
        return A


def manhattan(x1, x2):
    """ Computes the manhattan match-score on batches of vectors x1 and x2.

        Args:
            x1, x2: torch.Tensors of shape (batch_size, 1, input_size)
                The batches of vectors we are computing match-scores for.

        Returns
            scores: torch.Tensor of shape (batch_size, 1)
                The match-scores for the batches of vectors x1 and x2.
    """
    return 1.0 / (1.0 + torch.norm(x - y, p=1, dim=2))


def euclidean(x1, x2):
    """ Computes the euclidean match-score on batches of vectors x1 and x2.

        Args:
            x1, x2: torch.Tensors of shape (batch_size, 1, input_size)
                The batches of vectors we are computing match-scores for.

        Returns
            scores: torch.Tensor of shape (batch_size, 1)
                The match-scores for the batches of vectors x1 and x2.
    """
    return 1.0 / (1.0 + torch.norm(x1 - x2, p=2, dim=2))


def cosine(x1, x2):
    """ Computes the cosine match-score on batches of vectors x1 and x2.

        Args:
            x1, x2: torch.Tensors of shape (batch_size, 1, input_size)
                The batches of vectors we are computing match-scores for.

        Returns
            scores: torch.Tensor of shape (batch_size, 1)
                The match-scores for the batches of vectors x1 and x2.
    """
    dot_products = torch.bmm(x1, x2.permute(0, 2, 1)).squeeze(2)
    norm_x1 = torch.norm(x1, p=2, dim=2)
    norm_x2 = torch.norm(x2, p=2, dim=2)
    return dot_products / (norm_x1 * norm_x2)
