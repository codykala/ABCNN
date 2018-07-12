# coding=utf-8

import json
import os
import torch
import torch.nn as nn

from data.datasets import load_datasets
from embeddings.embeddings import create_embedding_matrix
from model.attention.abcnn1 import ABCNN1Attention
from model.attention.abcnn2 import ABCNN2Attention
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
    
    # Store output sizes for each Block
    # (needed for final layer initialization)
    output_sizes = [embeddings_size]  

    # Create the Blocks
    blocks = []
    for block in config["model"]["blocks"]:

        # Parameters common to all Blocks
        input_size = block["input_size"]
        output_size = block["output_size"]
        width = block["width"]

        if block["type"] == "bcnn":

            # Build the Block
            conv = Convolution(input_size, output_size, width, 1)
            pool = WidthAP(width)
            blocks.append(BCNNBlock(conv, pool))
            output_sizes.append(output_size)
        
        elif block["type"] == "abcnn1":

            # Parameters specific to ABCNN-1 Block
            match_score = block["match_score"]
            share_weights = block["share_weights"]

            # Build the Block
            attn = ABCNN1Attention(input_size, max_length, share_weights, match_score)
            conv = Convolution(input_size, output_size, width, 2)
            pool = WidthAP(width)
            blocks.append(ABCNN1Block(attn, conv, pool))
            output_sizes.append(output_size)
        
        elif block["type"] == "abcnn2":

            # Parameters specific to ABCNN-2 Block
            match_score = block["match_score"]

            # Build the block
            conv = Convolution(input_size, output_size, width, 1)
            attn = ABCNN2Attention(max_length, width, match_score)
            blocks.append(ABCNN2Block(conv, attn))
            output_sizes.append(output_size)

        elif block["type"] == "abcnn3":

            # Parameters specific to ABCNN-3 Block
            match_score = block["match_score"]
            share_weights = block["share_weights"]

            # Build the Block
            attn1 = ABCNN1Attention(input_size, max_length, share_weights, match_score)
            conv = Convolution(input_size, output_size, width, 2)
            attn2 = ABCNN2Attention(max_length, width, match_score)
            blocks.append(ABCNN3Block(attn1, conv, attn2))
            output_sizes.append(output_size)

        else:
            raise NotImplementedError

    # Compute size of final output layer
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
    elif classname.find("ABCNN1Attention") != -1:
        nn.init.xavier_normal_(m.W1)
        nn.init.xavier_normal_(m.W2)
