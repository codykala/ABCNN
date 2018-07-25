# coding=utf-8

import torch
import torch.nn as nn

from model.pooling.allap import AllAP

class Model(nn.Module):
    """ Extends on the model introduced in this paper:

        http://www.aclweb.org/anthology/Q16-1019

        Instead of using only a single window size for each convolutional layer,
        multiple window sizes are used. Different average pooling layers are used
        to shrink the outputs of these convolutional layers to their original sizes,
        and then these outputs are stacked and fed into the next layer.
    """

    def __init__(self, layers, use_all_layers, final_size):
        """ Initialize the ABCNN model layers.

            Args:
                embeddings: nn.Embedding Module
                    The embeddings matrix.
                layers: list of Layers modules
                    Contains the Layers of the CNN.
                use_all_layers: boolean
                    Specifies whether or not to foward the outputs of All-AP
                    layers applied to all non-final layers to form the final
                    representation.
                final_size: int
                    The number of inputs in the final fully connected layer
                    (accounts for both outputs from x1 and x2).

            Returns:
                None
        """
        super().__init__()   
        # self.embeddings = embeddings
        self.layers = nn.ModuleList(layers)
        self.use_all_layers = use_all_layers
        self.fc = nn.Linear(final_size, 2)
        self.ap = AllAP()


    def forward(self, inputs):
        """ Computes the forward pass over the network.

            Args:
                inputs: torch.Tensors of shape (batch_size, 2, max_length, embedding_size)
                    The initial feature maps for a batch of question pairs.

            Returns:
                out1, out2: torch.FloatTensors of shape (batch_size, 2)
                    The scores for each class for each pair of sequences.
        """
        # Collect all-ap outputs for each sequence
        outputs1 = []
        outputs2 = []

        # Extract the initial sequences
        x1 = inputs[:, 0, :, :].unsqueeze(1) # shape (batch_size, 1, max_length, embeddings_size)
        x2 = inputs[:, 1, :, :].unsqueeze(1) # shape (batch_size, 1, max_length, embeddings_size)

        # Store all-ap outputs for input layer
        a1, a2 = self.ap(x1), self.ap(x2) # shapes (batch_size, embeddings_size)
        outputs1.append(a1)
        outputs2.append(a2)

        # Process input through blocks
        for layer in self.layers:
            x1, x2, a1, a2 = layer(x1, x2)
            outputs1.append(a1)
            outputs1.append(a2)

        # Get final layer representation
        if self.use_all_layers:
            outputs = torch.cat(outputs1 + outputs2, dim=1)
        else:
            outputs = torch.cat([outputs1[-1], outputs2[-1]], dim=1)

        # Compute scores for each class
        logits = self.fc(outputs) # shape (batch_size, 2)
        return logits
