# coding=utf-8

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from .pooling.allap import AllAP


class Model(nn.Module):
    """ Implements the CNN models described in this paper:

        http://www.aclweb.org/anthology/Q16-1019
    """

    def __init__(self, embeddings, blocks):
        """ Initializes the ABCNN model layers.

            Args:
                embeddings: torch.nn.Embedding
                    Contains the word embeddings.
                blocks: list of Blocks
                    Contains Block modules defining the layers of the CNN.
                    
            Returns:
                None
        """
        super().__init__()
        self.embeddings = embeddings
        self.blocks = nn.ModuleList(blocks)

    def forward(self, idxs1, idxs2):
        """ Computes the forward pass over the network.

            Args:
                idxs1. idxs2: list of lists of ints

            Returns:
                out1, out2: torch.FloatTensors of shape (batch_size, ?) 
                    The output of the model for each pair of sequences.
        """
        # Store the outputs of the All-AP layers for each block and for each 
        # sequence
        out1 = []
        out2 = []

        # Forward pass
        x1 = self.embeddings(idxs1)
        x2 = self.embeddings(idxs2)
        for block in self.blocks:
            x1 = block(x1)
            x2 = block(x2)
            out1 += AllAP(x1)
            out2 += AllAP(x2)

        # Concatenate the All-AP layer outputs to get final representations
        out1 = torch.cat(out1, dim=1) 
        out2 = torch.cat(out2, dim=1)
        return out1, out2