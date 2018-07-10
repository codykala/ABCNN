# coding=utf-8

import numpy as np
import os
import pandas as pd
import re
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from tqdm import tqdm

from data.dataset import QuestionDataset

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
                The pre-trained word embeddings, if embeddings_path is provided.
                Otherwise, this is None.

        Returns:
            embeddings: nn.Embedding of shape (vocab_size, embeddings_size)
                The matrix of word embeddings.
    """
    # Pre-trained word vectors size must match given embeddings size
    if word2vec is not None:
        assert (word2vec.vector_size == embeddings_size)

    # Initialize the embedding matrix
    embeddings = 1 * np.random.randn(len(word2index) + 1, embeddings_size)
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


def convert_words_to_indices(datapaths, embeddings_path, max_length):
    """ Converts questions, which are represented as strings, to lists of
        indices into the embedding matrix. Additional, builds mappings from
        words to indices and vice versa.

        This code is based on code from Elior Cohen's MaLSTM notebook, which 
        can be found here:

        https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb  

        Args:
            datapaths: dict
                Each key is the name of the dataset and the value is the path 
                to the csv file containing a dataset. Each dataset is represented 
                as a pd.DataFrame organized into 3 columns: "question1", 
                "question2", and "is_duplicate". Each question is represented as a 
                string.
            embeddings_path: string
                Path to the file containing the pre-trained word embeddings.
            max_length: int
                The maximum length of sequences allowed. If a sequence
                is longer than the maximum length, then it is truncated.
                If a sequence is shorter than the maximum length, then it
                is padded with 0s.
        
        Returns:
            datasets: dict
                Each key is the name of the dataset and the value is a
                pd.DataFrame storing the dataset. Each dataset contains question
                pairs, and each question is represented as a list of int.
            word2index: dict
                Mapping from word/token to index in embeddings matrix.
            word2vec: gensim.models.Word2VecKeyedVectors or None
                The pre-trained word embeddings, if embeddings_path is provided.
                Otherwise, this is None.
    """
    # Read in datasets
    datasets = {name: pd.read_csv(datapath) for name, datapath in datapaths.items()}

    # Load pre-trained embeddings
    word2vec = None
    if os.path.isfile(embeddings_path):
        word2vec = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)

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
        # q2n -> question numbers representation
        q2n_rep = np.zeros((num_examples, 2, max_length)) 
        labels = np.array(dataset["is_duplicate"], dtype=int)
        print("Processing dataset: {}".format(name))
        print(q2n_rep.shape, labels.shape)
        with tqdm(total=num_examples) as pbar:
            for index, row in dataset.iterrows():

                # Iterate through the text of both questions of the row
                for i, question in enumerate(questions_cols):

                    q2n = []  
                    for word in text_to_word_list(row[question]):

                        # Check for unwanted words
                        if word2vec is not None:
                            # If a stop word has a pretrained word embedding, we
                            # should use that instead of a random embedding.
                            if word in stops and word not in word2vec.vocab:
                                continue
                        else:
                            if word in stops:
                                continue

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
        datasets[name] = QuestionDataset(q2n_rep, labels)

    return datasets, word2index, word2vec


def text_to_word_list(text):
    """ Preprocess and convert texts to a list of words.

        This code was taken from Elior Cohen's MaLSTM code, which can be found 
        here:

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