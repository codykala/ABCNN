# coding=utf-8

import json
import os
import torch
import torch.nn as nn

from data.datasets import load_datasets
from embeddings.embeddings import create_embedding_matrix
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
                Contains the information needed to initialize the
                datasets and model. See "config.json" for configuration
                details.
    """
    with open(config_path) as json_file:
        config = json.load(json_file)
        return config


def setup_datasets_and_model(config):
    """ Creates a CNN model using the given configuration.

        Args:
            config: dict
                Contains the information needed to initialize the datasets
                and model. See "config.json" for configuration details.

        Returns:
            model: nn.Module
                The instantiated model.
                
    """
    # Load the data
    data_paths = config["data_paths"]
    embeddings_path = config["embeddings"]["embeddings_path"]
    max_length = config["model"]["max_length"]
    datasets, word2index, word2vec = \
        load_datasets(data_paths, embeddings_path, max_length)

    # Create the embedding matrix
    embeddings_size = config["embeddings"]["embeddings_size"]
    embeddings = \
        create_embedding_matrix(embeddings_size, word2index, word2vec)    
    
    # Create the Blocks
    blocks = []
    output_sizes = [embeddings_size]
    for block in config["model"]["blocks"]:

        if block["type"] == "bcnn":
            
            conv = Convolution(**block["conv_params"])
            pool = WidthAP(**block["pool_params"])
            blocks.append(BCNNBlock(conv, pool))
            
            output_size = block["conv_params"]["output_size"]
            output_sizes.append(output_size)
        
        else:
            raise NotImplementedError

    # Compute size of output layer
    use_all_layers = config["model"]["use_all_layers"]
    final_size = sum(output_sizes) if use_all_layers else output_sizes[-1]

    # Create the model
    model = Model(embeddings, blocks, use_all_layers, final_size).float()
    model = model.cuda() if USE_CUDA else model
    model.apply(weights_init)
    return datasets, model

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.1)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.1)
