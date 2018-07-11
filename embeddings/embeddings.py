# coding=utf-8

import numpy as np
import os
import pandas as pd
import re
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from tqdm import tqdm

def create_embedding_matrix(embeddings_size, word2index, word2vec):
    """ Creates the embedding matrix. 

        This code is based on code from Elior Cohen's MaLSTM notebook,
        which can be found here:

        https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb 

        Args:
            embeddings_size: int
                The dimension of the word embeddings.
            word2index: dict
                Mapping from word/token to index in embeddings matrix.
            word2vec: gensim.models.Word2VecKeyedVectors or None
                Word2vec model storing the pre-trained word embeddings.
    
        Returns:
            embeddings: nn.Embedding of shape (vocab_size, embeddings_size)
                The matrix of word embeddings.
    """
    # Pre-trained word vectors size must match given embeddings size
    if word2vec is not None:
        assert (word2vec.vector_size == embeddings_size)

    # Initialize the embedding matrix
    # Embeddings initialized by drawing uniformly from [-0.01, 0.01]
    embeddings = \
        np.random.uniform(
            low=-0.01, 
            high=0.01, 
            size=(len(word2index) + 1, embeddings_size)
        )
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    # Words without pretrained embeddings will be assigned random embeddings
    if word2vec is not None:
        with tqdm(total=len(word2index)) as pbar:
            for word, index in word2index.items():
                if word in word2vec.vocab:
                    embeddings[index] = word2vec.word_vec(word) 
                pbar.update(1)

    # Convert embeddings to nn.Embeddings layer
    embeddings = nn.Embedding.from_pretrained(torch.from_numpy(embeddings))
    return embeddings