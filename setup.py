# coding=utf-8

import json
import numpy as np
import os
import re
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

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
            datasets: list of pd.DataFrame
                Each DataFrame is a dataset. Each dataset contains question
                pairs, and each question is represented as a list of indices
                into the embedding matrix. 
            model: nn.Module
                The instantiated model.
            word2index: dict
                Mapping from a word/token to its index in the embeddiing matrix.
            index2word: dict
                Mapping from index in the embedding matrix to its word/token. 
                
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
                include_all_pooling=config["include_all_pooling"])
    if USE_CUDA:
        model = model.cuda()
    return model