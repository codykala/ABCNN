# coding=utf-8

import torch
import torch.nn as nn

from model.pooling.allap import AllAP


class Model(nn.Module):
    """ Implements the CNN models described in this paper:

        http://www.aclweb.org/anthology/Q16-1019
    """

    def __init__(self, embeddings, blocks, use_all_layers, final_size):
        """ Initializes the ABCNN model layers.

            Args:
                embeddings: torch.nn.Embedding
                    Contains the word embeddings.
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
        self.embeddings = embeddings
        self.blocks = nn.ModuleList(blocks)
        self.use_all_layers = use_all_layers
        self.fc = nn.Linear(2 * final_size, 2)  
        self.similarity = nn.CosineSimilarity(dim=1)
        self.ap = AllAP() 
        
    def forward(self, idxs):
        """ Computes the forward pass over the network.

            Args:
                idxs: np.array of shape (batch_size, 2, max_length)
                    Contains the index representations for a pair of
                    sequences.

            Returns:
                out1, out2: torch.FloatTensors of shape (batch_size, 1) 
                    The output of the model for each pair of sequences.
        """
        # Collect all-ap outputs for each sequence
        outputs1 = []
        outputs2 = []

        # Forward pass
        x1 = self.embeddings(idxs[:, 0]).unsqueeze(1)  # shape (batch_size, 1, max_length, embedding_size)
        x2 = self.embeddings(idxs[:, 1]).unsqueeze(1)  

        # Store all-ap outputs for embedding layer
        a1, a2 = self.ap(x1), self.ap(x2) # shapes (batch_size, embedding_size)
        outputs1.append(a1)
        outputs2.append(a2)

        for block in self.blocks:

            # Compute input to next layer
            x1, a1 = block(x1) # shapes (batch_size, 1, max_length, height), (batch_size, height)
            x2, a2 = block(x2) 

            # Store all-ap outputs for current layer
            outputs1.append(a1)
            outputs2.append(a2)

        # Get final layer representation
        if self.use_all_layers:
            outputs = torch.cat(outputs1 + outputs2, dim=1)
        else:
            outputs = torch.cat([outputs1[-1], outputs2[-1]], dim=1)        

        logits = self.fc(outputs) # shape (batch_size, 2)
        return logits
