# coding=utf-8

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from pooling.allap import AllAP


class Model(nn.Module):
    """ Implements the CNN models described in this paper:

        http://www.aclweb.org/anthology/Q16-1019
    """

    def __init__(self, embeddings, blocks, include_all_pooling=True):
        """ Initializes the ABCNN model layers.

            Args:
                embeddings: torch.nn.Embedding
                    Contains the word embeddings.
                blocks: list of Blocks
                    Contains Block modules defining the layers of the CNN.
                include_all_pooling: boolean
                    Optional, specifies whether or not to forward the outputs 
                    of All-AP layers applied to all non-final Average Pooling 
                    layers to form the final representation. By default, this 
                    value is True.

            Returns:
                None
        """
        super().__init__()
        self.embeddings = embeddings
        self.blocks = nn.ModuleList(blocks)
        self.include_all_pooling = include_all_pooling

    def forward(self, idxs1, idxs2):
        """ Computes the forward pass over the network.

            Args:
                idxs1. idxs2: list of lists of ints
                    Each list is a list of row indices into the embedding
                    matrix.

            Returns:
                out1, out2: torch.FloatTensors of shape (batch_size, ?) 
                    The output of the model for each pair of sequences.
        """
        # Store the outputs of the All-AP layers for each input
        if self.include_all_pooling:
            outputs1 = []
            outputs2 = []

        # Forward pass
        x1 = self.embeddings(idxs1)
        x2 = self.embeddings(idxs2)
        for block in self.blocks:
            x1, out1 = block(x1)
            x2, out2 = block(x2)
            if self.include_all_pooling:
                outputs1 += out1
                outputs2 += out2

        # Concatenate the All-AP layer outputs to get final representations
        if self.include_all_pooling:
            out1 = torch.cat(outputs1, dim=1) 
            out2 = torch.cat(outputs2, dim=1)
        # Otherwise, only use output of final All-AP layer as final representation
        return out1, out2