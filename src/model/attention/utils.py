# coding=utf-8

import torch

# Use GPU if available, otherwise use CPU
USE_CUDA = torch.cuda.is_available()

def compute_attention_matrix(x1, x2, match_score):
    """ Computes the attention feature map for the batch of inputs x1 and x2.

        The following description is taken directly from the ABCNN paper:

        Let F_{i, r} in R^{d x s} be the representation feature map of
        sentence i (i in {0, 1}). Then we define the attention matrix A in R^{s x s}
        as follows:

            A_{i, j} = match-score(F_{0, r}[:, i], F_{1, r}[:, j])

        Args:
            x1, x2: torch.Tensors of shape (batch_size, 1, max_length, input_size)
                A batch of input tensors.
            match_score: function
                The match-score function to use.

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
            A[:, :, i, j] = match_score(b1, b2)
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
    return 1.0 / (1.0 + torch.norm(x1 - x2, p=1, dim=2))


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
