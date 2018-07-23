# coding=utf-8

import torch
import torch.nn as nn

from model.pooling.allap import AllAP


class Model(nn.Module):
    """ Implements the CNN models described in this paper:

        http://www.aclweb.org/anthology/Q16-1019
    """

    def __init__(self, blocks, use_all_layers, final_size):
        """ Initializes the ABCNN model layers.

            Args:
                blocks: list of Blocks Module
                    Contains Block modules defining the layers of the CNN.
                include_all_pooling: boolean
                    Specifies whether or not to forward the outputs 
                    of All-AP layers applied to all non-final Average Pooling 
                    layers to form the final representation.
                final_size: int
                    The number of nodes in the output layer.

            Returns:
                None
        """
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.use_all_layers = use_all_layers
        self.fc = nn.Linear(2 * final_size, 2)  
        self.ap = AllAP() 
        
    def forward(self, inputs):
        """ Computes the forward pass over the network.

            Args:
                inputs: torch.Tensors of shape (batch_size, 2, max_length, embedding_size)
                    The initial feature maps for a batch of question pairs.

            Returns:
                out1, out2: torch.FloatTensors of shape (batch_size, 1) 
                    The output of the model for each pair of sequences.
        """
        # Collect all-ap outputs for each sequence
        outputs1 = []
        outputs2 = []

        # Extract the sequences
        x1 = inputs[:, 0, :, :].unsqueeze(1) # shape (batch_size, 1, max_length, embedding_size)
        x2 = inputs[:, 1, :, :].unsqueeze(1) # shape (batch_size, 1, max_length, embedding_size)

        # Store all-ap outputs for embedding layer
        a1, a2 = self.ap(x1), self.ap(x2) # shapes (batch_size, embedding_size)
        outputs1.append(a1)
        outputs2.append(a2)

        # Process input through blocks
        for block in self.blocks:
            x1, x2, a1, a2 = block(x1, x2)
            outputs1.append(a1)
            outputs2.append(a2)

        # Get final layer representation
        if self.use_all_layers:
            outputs = torch.cat(outputs1 + outputs2, dim=1)
        else:
            outputs = torch.cat([outputs1[-1], outputs2[-1]], dim=1)        

        logits = self.fc(outputs) # shape (batch_size, 2)
        return logits
