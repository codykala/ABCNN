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
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from tqdm import tqdm

from model.attention.abcnn1 import ABCNN1Attention
from model.attention.abcnn2 import ABCNN2Attention
from model.blocks.abcnn1 import ABCNN1Block
from model.blocks.abcnn2 import ABCNN2Block
from model.blocks.abcnn3 import ABCNN3Block
from model.blocks.bcnn import BCNNBlock
from model.convolution.conv import Convolution
from model.model import Model
from model.pooling.allap import AllAP
from model.pooling.widthap import WidthAP
from process import setup_dataset

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
    # Read in relevant config info
    data_paths = config["data_paths"]
    embeddings_size = config["embeddings"]["size"]
    max_length = config["model"]["max_length"]

    # Setup the datasets
    datasets = {name: pd.read_csv(data_path) for name, data_path in data_paths.items()}
    word_vectors = setup_word_vectors(config)
    for name, dataset in datasets.items():
        dataset, _ = setup_dataset(dataset, word_vectors, embeddings_size, max_length)
        datasets[name] = dataset

    # Setup the model
    model = setup_model(config)
    return datasets, model


def setup_model(config):
    """ Sets up the model for training/evaluation.

        Args:
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
    model = Model(blocks, use_all_layers, final_size).float()
    model = model.cuda() if USE_CUDA else model
    model.apply(weights_init)
    return model


def setup_word_vectors(config):
    """ Loads the pre-trained word vectors. The word vector file can be in
        Word2Vec or FastText formats.

        Args:
            config: dict
                Contains the information needed to initialize the embeddings
                model.

        Returns:
            word_vectors: KeyedVectors, FastTextKeyedVectors, or None
                The pretrained word embeddings. If the embeddings path
                is for a pre-trained Word2Vec model, then a KeyedVectors
                instance is returned. If the embeddings path is for a
                pre-trained FastText model, then a FastTextKeyedVectors
                instance is returned. Otherwise, None is returned.
    """
    # Get relevant parameters from config file
    embeddings_path = config["embeddings"]["path"]
    embeddings_format = config["embeddings"]["format"]
    is_binary = config["embeddings"]["is_binary"]
   
    # Load pre-trained word embeddings
    word_vectors = None
    if embeddings_format == "word2vec":
        if os.path.isfile(embeddings_path):
            print("Loading Word2Vec word vectors from: {}".format(embeddings_path))
            word_vectors = KeyedVectors.load_word2vec_format(embeddings_path, binary=is_binary)
    elif embeddings_format == "fasttext":
        if os.path.isfile(embeddings_path):
            print("Loading FastText word vectors from: {}".format(embeddings_path))
            embeddings_model = FastText.load_fasttext_format(embeddings_path)
            word_vectors = embeddings_model.wv 
    else:
        raise Exception("Unsupported type. Must be one of 'word2vec' or 'fasttext'.")
    return word_vectors


def setup_dataset(examples, word_vectors, embeddings_size, max_length):
    """ Convert question pairs and labels into machine-readable datasets that
        can be used for training and evaluation.

        Args:
            examples: pd.DataFrame or list of lists
                Contains the examples. If examples is a pd.Dataframe instance,
                then the examples must be separated into columns named
                "question1", "question2", and "is_duplicate". If examples is a
                list of lists, then each example will be in the format 
                [question1, question2, label]. Each question is a string of words, 
                and the label is 0 or 1.
            word_vectors: KeyedVectors, FastTextKeyedVectors, or None
                The pre-trained word vectors. If word_vectors is a KeyedVectors
                instance, then only words in its vocabulary can be used.
                If word_vectors is a FastTextKeyedVectors instance, then OOV
                words can potentially be generated on the fly using n-gram word
                vectors. If word_vectors is None, then a random embedding is used.
            embeddings_size: int
                The dimension of the word embeddings
            max_length: int
                The maximum length of questions/sequences.

        Returns:
            dataset: TensorDataset
                Contains the feature maps and labels for all of the question pairs
                in the given set of examples.
            examples: list of tuples
                Contains the examples in the format (question1, question2, label)
                Each question is represented as a list of words/tokens. The
                label is a 0 or a 1 (integer). Padding tokens (resp. features) are
                added to the examples (resp. feature maps) so that each question
                has max_length tokens.
    """
    

    # Parse the data into the same format before processing
    if isinstance(examples, pd.DataFrame):
        pairs = examples[["question1", "question2"]].values.tolist()
        labels = examples["is_duplicate"].tolist()
    elif isinstance(examples, list):
        pairs = [(x[0], x[1]) for x in examples] 
        labels = [int(x[2]) for x in examples]

    # Convert to tensors / clean up text
    features, texts = texts_to_features(pairs, word_vectors, embeddings_size, max_length)
    examples = [(text[0], text[1], label) for text, label in zip(texts, labels)]
    labels = torch.LongTensor(labels)
    dataset = TensorDataset(features, labels)
    return dataset, examples


def texts_to_features(pairs, word_vectors, embeddings_size, max_length):
    """ Converts a list of texts (strings) into their feature maps.

        Args:
            texts: list of tuple of  string
                The text we would like to featurize in the format (question1, question2).
            word_vectors: KeyedVectors, FastTextKeyedVectors, or None
                The pre-trained word vectors. If word_vectors is a KeyedVectors
                instance, then only words in its vocabulary can be used.
                If word_vectors is a FastTextKeyedVectors instance, then OOV
                words can potentially be generated on the fly using n-gram word
                vectors. If word_vectors is None, then a random embedding is used.
            embeddings_size: int
                The dimension of the word embeddings
            max_length: int
                The maximum length of questions/sequences.

        Returns:
            feature_maps: torch.Tensor of shape (batch_size, max_length, embedding_size)
                The feature maps for each text.
            processed_texts: list of lists
                Contains the processed text for each question. Stop words are removed
                and <unk> tokens are added to pad the sequence to max length.
    """
    # Process texts
    feature_maps = []
    processed_texts = []
    for pair in tqdm(pairs):

        # Process each question separately
        feature_map = []
        processed_text = []
        for question in pair:
        
            # Parse text into words and remove stop words
            words = text_to_word_list(question)
            words = remove_stop_words(words, word_vectors)

            # Convert words to features
            # If a word has no word vector, use a random word vector
            features = []
            for word in words:
                try:
                    feature = torch.from_numpy(word_vectors[word])
                except (RuntimeError, KeyError):
                    feature = torch.Tensor(embeddings_size).uniform_(-0.01, 0.01)
                feature = feature.unsqueeze(0) # shape (1, embeddings_size)
                features.append(feature)

            # Truncate if necessary
            length = len(words) if len(words) < max_length else max_length
            features = features[:length]
            words = words[:length]

            # Add padding if necessary
            if length < max_length:
                num_padding = max_length - length
                padding = torch.zeros(num_padding, embeddings_size)
                unks = ["<unk>"] * num_padding
                features.append(padding)
                words.extend(unks)

            # Combine features into single feature map
            features = torch.cat(features, dim=0) # shape (max_length, embedding_size)
            feature_map.append(features)
            processed_text.append(words)

        # Stack feature maps for the question pair
        feature_map = torch.stack(feature_map) # shape (2, batch_size, embedding_size)
        feature_maps.append(feature_map)
        processed_texts.append(processed_text)

    # Stack feature maps for all examples
    feature_maps = torch.stack(feature_maps)
    return feature_maps, processed_texts


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


def remove_stop_words(words, word_vectors):
    """ Removes all of the stop words.

        Args:
            words: list of string
                The words in the text.
            word_vectors: KeyedVectors, FastTextKeyedVectors, or None
                The pre-trained word vectors. If word_vectors is a KeyedVectors
                instance, then only words in its vocabulary can be used.
                If word_vectors is a FastTextKeyedVectors instance, then OOV
                words can potentially be generated on the fly using n-gram word
                vectors. If word_vectors is None, then a random embedding is used.
        
        Returns:
            words: list of string
                The words in the text with stop words removed.
    """
    stops = set(stopwords.words("english"))
    return list(filter(lambda w: w not in stops or w in word_vectors.vocab, words))


def setup_embedding_matrix(word2index, word_vectors, config):
    """ Creates the embedding matrix. 

        This code is based on code from Elior Cohen's MaLSTM notebook,
        which can be found here:

        https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb 

        Args:
            word2index: dict
                Mapping from word/token to index in embeddings matrix.
            word_vectors: KeyedVectors or None
                The pretrained word embeddings. If a pre-trained
                model was provided, then this should be a KeyedVectors
                instance. Otherwise, this should be None.
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
    if word_vectors is not None:
        with tqdm(total=num_words) as pbar:
            for word, index in word2index.items():
                try:
                    embeddings[index] = word_vectors[word]
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
        raise Exception("Unsupported type. Must be one of 'bcnn', 'abcnn1', 'abcnn2', 'abcnn3'.")

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
