# coding=utf-8

import pandas as pd
import re
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from torch.utils.data import TensorDataset
from tqdm import tqdm

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
            texts: list of tuple of string
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
            feature_maps: torch.Tensor of shape (num_examples, max_length, embedding_size)
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
        
if __name__ == "__main__":

    from gensim.models import FastText
    from gensim.test.utils import common_texts
    from nltk.corpus import stopwords
    
    from setup import read_config
    from setup import setup_word2vec_model

    config = read_config("config.json")
    embeddings_size = config["embeddings"]["size"]
    max_length = config["model"]["max_length"]
    
    # word_vectors = setup_word2vec_model(config)
    model = FastText(common_texts, size=300, window=3, min_count=1, iter=10)
    word_vectors = model.wv

    # examples = pd.read_csv("data/hulo/train.csv")
    # print(examples.iloc[[0]])

    examples = [
        ["how do I find a printer?", "The printer is not working. Looks to be offline.", 1],
        ["How to connect to printer?", "I can't add the printer", 1],
        ["How do I fix the keyboard of my mac", "VPN connection fail", 0],
        ["My VPN is broken", "Can't log in with VPN", 1]
    ]
    print(examples[0])

    dataset, examples = setup_dataset(examples, word_vectors, embeddings_size, max_length)
    print(examples[0])
