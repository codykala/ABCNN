# coding=utf-8

import json
import numpy as np
import os
import pandas as pd
import re
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from gensim.models import FastText
from nltk.corpus import stopwords
from torch.utils.data import TensorDataset
from tqdm import tqdm

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


def setup(config):
    """ Handles all of the setup needed to run an ABCNN model.

        Args:
            config: dict
                Contains the information needed to initialize the datasets
                and model. See "config.json" for configuration details.

        Returns:
            datasets: dict of TensorDatasets
                 Contains the datasets to be used for training/evaluation.
            model: nn.Module
                The instantiated model.
    """
    embeddings_wv = setup_word2vec_model(config)
    datasets, word2index = setup_datasets(embeddings_wv, config)
    embeddings = setup_embedding_matrix(word2index, embeddings_wv, config)    
    model = setup_model(embeddings, config)
    return datasets, model


def setup_model(embeddings, config):
    """ Sets up the model for training/evaluation.

        Args:
            embeddings: nn.Embedding
                The matrix of word embeddings.
            config: dict
                Contains the information needed to setup the model.

        Returns:
            model: nn.Module
                The instantiated model.
    """
    print("Creating the ABCNN model...")

    # Get relevant parameters
    embeddings_size = config["embeddings"]["size"]
    max_length = config["model"]["max_length"]
    block_configs = config["model"]["blocks"]
    use_all_layers = config["model"]["use_all_layers"]

    # Create blocks
    blocks = []
    output_sizes = [embeddings_size]
    for block_config in block_configs:
        block, output_size = setup_block(max_length, block_config)
        blocks.append(block)
        output_sizes.append(output_size)

    # Compute the size of the FC layer
    final_size = sum(output_sizes) if use_all_layers else output_sizes[-1]

    # Put it all together
    model = Model(embeddings, blocks, use_all_layers, final_size).float()
    model = model.cuda() if USE_CUDA else model
    model.apply(weights_init)
    return model


def setup_word2vec_model(config):
    """ Loads the pre-trained word embedding model from file. The word 
        embedding model can be either Word2Vec or FastText.

        Args:
            config: dict
                Contains the information needed to initialize the embeddings
                model.

        Returns:
            embeddings_wv: KeyedVectors or None
                The pretrained word embedding model. If a pre-trained model
                is provided, then a KeyedVectors instance is returned.
                Otherwise, None is returned.
    """
    print("Loading pre-trained word embedding model...")
    
    # Get relevant parameters from config file
    embeddings_path = config["embeddings"]["path"]
    embeddings_format = config["embeddings"]["format"]
    is_binary = config["embeddings"]["is_binary"]
   
    # Load pre-trained word embeddings
    embeddings_wv = None
    if embeddings_format == "word2vec":
        if os.path.isfile(embeddings_path):
            embeddings_wv = KeyedVectors.load_word2vec_format(embeddings_path, binary=is_binary)
    elif embeddings_format == "fasttext":
        if os.path.isfile(embeddings_path):
            embeddings_model = FastText.load_fasttext_format(embeddings_path)
            embeddings_wv = embeddings_model.wv
    else:
        raise Exception("Unsupported type. Must be one of 'word2vec' or 'fasttext'.")
    return embeddings_wv


def setup_datasets(embeddings_wv, config):
    """ Converts questions, which are represented as strings, to lists of
        indices into the embedding matrix. Additionally builds mappings from
        words to indices and vice versa.

        This code is based on code from Elior Cohen's MaLSTM notebook, which
        can be found here:
        
        https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb
        
        Args:
            embeddings_wv: KeyedVectors or None
                The pre-trained word embeddings. If a pre-trained word embeddings
                model is provided, then a KeyedVectors instance is returned.
                Otherwise, None is returned.
            config: dict
                Contains the information needed to initialize the datasets.
                See "config.json" for configuration details.

        Returns:
            datasets: dict
                Each key is the name of the dataset and the value is a
                pd.DataFrame storing the dataset. Each dataset contains question
                pairs, and each question is represented as a list of int.
            word2index: dict
                Mapping from word/token to index in embeddings matrix.
    """
    print("Setting up datasets...")
    
    # Read in relevant parameters
    datapaths = config["data_paths"]
    embeddings_path = config["embeddings"]["path"]
    max_length = config["model"]["max_length"]

    # Read in datasets
    datasets = {name: pd.read_csv(datapath) for name, datapath in datapaths.items()}

    # Load stop words
    stops = set(stopwords.words('english'))

    # Create word-index mappings
    # '<unk>' is never used, only a stand-in for the [0, 0, ..., 0] embedding
    word2index = dict()
    index2word = {"<unk>": 0}

    # Iterate over the questions in each dataset
    questions_cols = ['question1', 'question2']
    for name, dataset in datasets.items():
        num_examples = len(dataset)
        q2n_rep = np.zeros((num_examples, 2, max_length))
        labels = np.array(dataset["is_duplicate"], dtype=int)
        with tqdm(total=num_examples) as pbar:
            for index, row in dataset.iterrows():

                # Iterate through the text of both questions of the row
                for i, question in enumerate(questions_cols):

                    q2n = []  # q2n -> question numbers representation
                    for word in text_to_word_list(row[question]):

                        # Check for stop words
                        # For Word2Vec, use random embeddings for OOV
                        # For FastText, use n-gram embeddings for OOV before
                        # defaulting to random embeddings
                        if embeddings_wv is not None:
                            if word in stops and word not in embeddings_wv.vocab:
                                continue
                        else:
                            if word in stops:
                                continue

                        # Update word-index and q2n 
                        # Only non-stop words should make it here
                        if word not in word2index:
                            word2index[word] = len(index2word)
                            q2n.append(len(index2word))
                            index2word[len(index2word)] = word
                        else:
                            q2n.append(word2index[word])

                    # Add padding or truncate
                    length = max_length if len(q2n) > max_length else len(q2n)
                    q2n = q2n[:length]
                    q2n_rep[index, i, :length] = np.array(q2n, dtype=int)

                pbar.update(1)

        # Convert to LongTensors
        q2n_rep = torch.LongTensor(q2n_rep)
        labels = torch.LongTensor(labels) 
        datasets[name] = TensorDataset(q2n_rep, labels)

    return datasets, word2index


def text_to_word_list(text):
    """ Preprocess and convert texts to a list of words. This code was taken 
        from Elior Cohen's MaLSTM code, which can be found here:

        https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb
        
        Args:
            text: string
                The text to parse.

        Returns:
            text: list of string
                The parsed text.
    """
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


def setup_embedding_matrix(word2index, embeddings_wv, config):
    """ Creates the embedding matrix. 

        This code is based on code from Elior Cohen's MaLSTM notebook,
        which can be found here:

        https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb 

        Args:
            word2index: dict
                Mapping from word/token to index in embeddings matrix.
            embeddings_wv: KeyedVectors or None
                The pretrained word embedding model. If a pre-trained
                model is provided, then a KeyedVectors instance is
                returned. Otherwise, None is returned.
            config: dict
                Contains the information needed to initialize the model.

        Returns:
            embeddings: nn.Embedding of shape (vocab_size, embeddings_size)
                The matrix of word embeddings.
    """
    print("Creating embedding matrix...")
    
    # Get relevant parameters
    embeddings_size = config["embeddings"]["size"]
    num_words = len(word2index)

    # Initialize the embedding matrix
    # Words without pre-trained embeddings will be assigned random embeddings
    embeddings = np.random.uniform(-0.01, 0.01, (num_words + 1, embeddings_size))
    embeddings[0] = 0  # So that the padding will be ignored

    # Load in the pre-trained word embeddings
    if embeddings_wv is not None:
        with tqdm(total=num_words) as pbar:
            for word, index in word2index.items():
                try:
                    embeddings[index] = embeddings_wv[word]
                except KeyError:
                    pass
                pbar.update(1)

    # Convert to an nn.Embedding layer
    embeddings = nn.Embedding.from_pretrained(torch.from_numpy(embeddings))
    return embeddings


def setup_block(max_length, block_config):
    """ Creates a single block for the CNN model.

        Args:
            max_length: int
                The maximum length for each sequence/question.
            block_config: dict
                Contains the information needed to create the block.

        Returns:
            block: Block module
                The desired Block module.
    """
    input_size = block_config["input_size"]
    output_size = block_config["output_size"]
    width = block_config["width"]
    dropout_rate = block_config["dropout_rate"]
    match_score = block_config["match_score"]
    share_weights = block_config["share_weights"]

    if block_config["type"] == "bcnn":
        conv = Convolution(input_size, output_size, width, 1)
        pool = WidthAP(width)
        block = BCNNBlock(conv, pool, dropout_rate=dropout_rate)
    
    elif block_config["type"] == "abcnn1":
        attn = ABCNN1Attention(input_size, max_length, share_weights, match_score)
        conv = Convolution(input_size, output_size, width, 2)
        pool = WidthAP(width)
        block = ABCNN1Block(attn, conv, pool, dropout_rate=dropout_rate)
    
    elif block_config["type"] == "abcnn2":
        conv = Convolution(input_size, output_size, width, 1)
        attn = ABCNN2Attention(max_length, width, match_score)
        block = ABCNN2Block(conv, attn, dropout_rate=dropout_rate)
    
    elif block_config["type"] == "abcnn3":
        attn1 = ABCNN1Attention(input_size, max_length, share_weights, match_score)
        conv = Convolution(input_size, output_size, width, 2)
        attn2 = ABCNN2Attention(max_length, width, match_score)
        block = ABCNN3Block(attn1, conv, attn2, dropout_rate=dropout_rate)

    else:
        raise Exception("Unexpected value of type. Should be one of 'bcnn', 'abcnn1', 'abcnn2', 'abcnn3'.")

    return block, output_size    


def weights_init(m):
    """ Initializes the weights for the modules in the CNN model. This function 
        is applied recursively to all modules in the model via the "apply"
        function.

        Args:
            m: nn.Module
                The module to initialize.
            
        Returns:
            None
    """ 
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
