# coding=utf-8

import torch
import torch.nn as nn

from model.pooling.allap import AllAP

class CNNLayer(nn.Module):
    """ Extends on the single Block module concept described in this paper: 

        http://www.aclweb.org/anthology/Q16-1019

        Instead of using a single convolutional layer followed by a single pooling
        layer, we use an ensemble of convolutional layers with different window sizes
        followed by pooling layers with corresponding widths. The outputs of these
        pooling layers are combined to form the input to the following section of
        the network.
    """

    def __init__(self, blocks, dropout_rate=0.5):
        """ Initializes the CNNLayer.

            Args:
                blocks: list of Block modules
                    Contains Block modules with different window sizes and 
                    pooling widths.
                dropout_rate: float
                    The probability that individual neurons are dropped out 
                    from the network. Must be a float between 0 and 1. By 
                    default, this value is 0.5.

            Returns:
                None
        """
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.ap = AllAP()
        

    def forward(self, x1, x2):
        """ Computes the forward pass over the BCNN Layer.
            
            Args:
                x1, x2: torch.FloatTensors of shape (batch_size, 1, max_length, input_size)
                    The inputs to the BCNN Block.

            Returns:
                w1, w2: torch.FloatTensors of shape (batch_size, 1, max_length, output_size)
                    The outputs of the w-ap Average Pooling layer. These are passed to
                    the next Block.
                a1, a2: torch.FloatTensors of shape (batch_size, output_size)
                    The outputs of the all-ap Average Pooling layer. These are optionally
                    passed to the output layer.
        """
        # Collect the outputs for the average and all-pooling layers
        # for each conv-pool combination.
        wap1, wap2 = [], []
        allap1, allap2 = [], []

        # Process inputs
        for block in self.blocks:
            w1, w2, a1, a2 = block(x1, x2)
            wap1.append(w1)
            wap2.append(w2)
            allap1.append(a1)
            allap2.append(a2)

        # Combine the outputs
        w1 = torch.cat(wap1, dim=3) # shape (batch_size, 1, max_length, output_size)
        w2 = torch.cat(wap2, dim=3)
        a1 = torch.cat(allap1, dim=1) # shape (batch_size, output_size)
        a2 = torch.cat(allap2, dim=1)
        return w1, w2, a1, a2
