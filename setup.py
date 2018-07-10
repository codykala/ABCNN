# coding=utf-8

import json
import os
import torch
import torch.nn as nn

from model.blocks.bcnn import BCNNBlock
from model.blocks.abcnn1 import ABCNN1Block
from model.blocks.abcnn2 import ABCNN2Block
from model.blocks.abcnn3 import ABCNN3Block
from model.convolution.conv import Convolution
from model.model import Model
from model.pooling.allap import AllAP
from model.pooling.widthap import WidthAP

# Use GPU if available, otherwise use CPU
USE_CUDA = torch.cuda.is_available()

def read_config(config_path):
    """ Reads in the configuration file from the given path.

        Args:
            config_path: string
                The path to the configuration file.

        Returns:
            config: dict
                Contains the information needed to initialize the layers
                of the Model.
    """
    with open(config_path) as json_file:
        config = json.load(json_file)
        return config


def setup_model(embeddings, config):
    """ Creates a CNN model using the given configuration.

        Args:
            embeddings: nn.Embedding of shape (vocab_size, embeddings_size)
                The matrix of word embeddings.
            config: dict
                Contains the information needed to initialize the layers
                of the Model. See config.json for configuration details.

        Returns:
            model: nn.Module
                The instantiated model.
                
    """
    # Create the Blocks
    blocks = []
    block_config = config["blocks"]
    for params in block_config:
        if params["type"] == "bcnn":
            conv = Convolution(**params["conv"])
            pool = WidthAP(**params["pool"])
            blocks.append(BCNNBlock(conv, pool))
        else:
            raise NotImplementedError

    # Create the model
    model = Model(embeddings, blocks, 
                include_all_pooling=config["include_all_pooling"]).float()
    if USE_CUDA:
        model = model.cuda()
    model.apply(weights_init)
    return model


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(m.weight)
    elif classname.find("Linear") != -1:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.1)
