# ABCNN

Implements the ABCNN model presented in the paper "ABCNN: Attention-Based
Convolutional Neural Network for Modeling Sentence Pairs" by Yin, Schutze,
Xiang, and Zhou (arXiv link: `https://arxiv.org/pdf/1512.05193.pdf`).

# Setup

Firstr, create a virtual environment and install the required dependencies
from `requirements.txt`. Make sure the virtual environment uses `python3`.
If you are using virtualenvwrapper, you can use the following commands:

```
mkvirtualenv --python=python3 <name-of-env>
pip3 install -r requirements.txt
```

You may also need to install `tkinter`. On a linux machine, you can do this
using the following command:

```
sudo apt-get install python3-tk
```

In order to train the model, you will need to have some pre-trained 
word embeddings (either word2vec or fasttext). You can use these commands
to download these publicly available word embeddings:

## FastText Embeddings

`wiki-news-300d-1M.vec.zip`: 1 million word vectors trained on Wikipedia 2017,
UMBC webbase corpus and statmt.org news dataset (16B tokens):

```
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip
```

`wiki-news-300d-1M-subword.vec.zip`: 1 million word vectors trained with subword 
infomation on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset 
(16B tokens).

```
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M-subword.vec.zip
```

`crawl-300d-2M.vec.zip`: 2 million word vectors trained on Common Crawl (600B tokens).

```
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip
```

## Word2Vec Embeddings

`GoogleNews-vectors-negative300.bin.gz`: 3 million word vectors trained on the
Google News dataset (100B tokens).

```
wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
```

# Usage

First, enter your virtual environment. If you are using `virtualenvwrapper`,
you can use the following command:

```
workon <name-of-env>
```

Before using the scripts, you may need to change the config file. Here is a
brief description of each key in the `config.json` file:

```
train_set: string 
    The name of the file containing the training examples.
val_set: string
    The name of the file containing the validation examples.
test_set: string
    The name of the file containing the test examples.
data_paths: dict
	Maps the names of each dataset to its filepath.
embeddings: dict
    Contains config settings for generating the emebdding matrix.
    The model can be trained using either word2vec or fasttext
    embeddings, but only word2vec models can be loaded from
    binary files. The `size` refers to the dimension of the
    word embeddings.
optim: dict
    Contains config settings for the optimizer. 
model: dict
	Contains config settings for the ABCNN model. Notably,
	the model is defined via the config settings specified
	under the "layers" key.
train: dict  
    Contains config settings for the training harness.
```

Once the changes to the `config.json` file are made, you will need to change
the shell scripts `train.sh` and `eval.sh` to use your config file. You may
also need to make the shell scripts executable. This can be done with the
following command:

```
chmod +x <name-of-file>
```

Once this is done, training can be performed using the following command:

```
./train.sh
```

Similarly, evaluation can be performed using the following command:

```
./eval.sh
```

To get more information about the command line arguments, you can use the
following command:

```
python src/main.py --help
```
