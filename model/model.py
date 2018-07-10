# coding=utf-8

import torch
import torch.nn as nn

from model.pooling.allap import AllAP


class Model(nn.Module):
    """ Implements the CNN models described in this paper:

        http://www.aclweb.org/anthology/Q16-1019
    """

    def __init__(self, embeddings, blocks, include_all_pooling=True):
        """ Initializes the ABCNN model layers.

            Args:
                embeddings: torch.nn.Embedding
                    Contains the word embeddings.
                blocks: list of Blocks Module
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

        self.num_blocks = len(blocks)
        self.similarity = nn.CosineSimilarity(dim=1)
        self.ap = AllAP() 
        
        if include_all_pooling:
            self.fc = nn.Linear(self.num_blocks + 1, 1)
        

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
        # Store the outputs of the All-AP layers for each input
        scores = []

        # Forward pass
        x1 = self.embeddings(idxs[:, 0]).unsqueeze(1)  # shape (batch_size, 1, max_length, embedding_size)
        x2 = self.embeddings(idxs[:, 1]).unsqueeze(1)  

        # Scores for input layer
        a1, a2 = self.ap(x1), self.ap(x2) # shapes (batch_size, embedding_size)
        sim = self.similarity(a1, a2).unsqueeze(1) # shape (batch_size, 1)
        scores.append(sim)

        for block in self.blocks:

            # Compute input to next layer
            x1, a1 = block(x1) # shapes (batch_size, 1, max_length, height), (batch_size, height)
            x2, a2 = block(x2) 

            # Compute score for current layer
            sim = self.similarity(a1, a2).unsqueeze(1) # shape (batch_size, 1)
            scores.append(sim) 

        # Return weighted sum of scores across all layers
        if self.include_all_pooling:
            scores = torch.cat(scores, dim=1) # shape (batch_size, num_blocks + 1)
            output = self.fc(scores).squeeze() # shape (batch_size,)
            return output

        # Return scores from final layer only
        output = scores[-1].squeeze() # shape (batch_size,)
        return output
