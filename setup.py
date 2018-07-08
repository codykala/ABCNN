# coding=utf-8

import torch
import torch.nn as nn
import json
from gensim.models import KeyedVectors

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


def load_word2vec_embeddings(embeddings_path):
    """ Loads the pre-trained word2vec word embeddings and generates
        the mapping and inverse mapping between words/tokens and
        their indices into the embedding matrix.

        Note: The embeddings will not be updated during the learning process.

        Args:
            embeddings_path: string
                Path to the file containing the pre-trained word embeddings.

        Returns:
            embeddings: torch.nn.Embedding of shape (vocab_size, embeds_size) 
                Contains the pre-trained word embeddings.
            word2index: dict
                Mapping from a word/token to its index in the embeddiing matrix.
            index2word: dict
                Mapping from index in the embedding matrix to its word/token. 
    """
    model = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
    weights = torch.FloatTensor(model.syn0)
    embeddings = nn.Embedding.from_pretrained(weights) 
    index2word = {i: k for i, k in enumerate(model.index2word)}
    word2index = {k: i for i, k in enumerate(model.index2word)}
    return embeddings, index2word, word2index


def setup_model(config):
    """ Creates a CNN model using the given configuration.

        Args:
            config: dict
                Contains the information needed to initialize the layers
                of the Model. See config.json for configuration details.

        Returns: 
            model: nn.Module
                The instantiated model.
            word2index: dict
                Mapping from a word/token to its index in the embeddiing matrix.
            index2word: dict
                Mapping from index in the embedding matrix to its word/token. 
                
    """
    # Parse the config dict
    embed_config = config["embeddings"]
    block_config = config["blocks"]

    # Load or create embeddings
    if embed_config["use_pretrained_embeddings"]:
        print("Loading pretrained embeddings...")
        embeddings, index2word, word2index = \
            load_word2vec_embeddings(embed_config["embeddings_path"])
        print("... Done.")
    else:
        raise NotImplementedError

    # Create the Blocks
    blocks = []
    for params in block_config:
        if params["type"] == "bcnn":
            conv = Convolution(**params["conv"])
            pool = WidthAP(**params["pool"])
            blocks.append(BCNNBlock(conv, pool))
        else:
            raise NotImplementedError

    # Create the model
    model = Model(embeddings, blocks)
    if USE_CUDA:
        model = model.cuda()
    return model, index2word, word2index
    

if __name__ == "__main__":
    config = read_config("config.json")
    model, index2word, word2index = setup_model(config)