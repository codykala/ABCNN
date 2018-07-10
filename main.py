# coding=utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

from embeddings.embeddings import load_datasets
from embeddings.embeddings import create_embedding_matrix
from setup import read_config
from setup import setup_model
from train import train

torch.set_default_tensor_type('torch.FloatTensor')

# Get configurations
config = read_config("config.json")
max_length = config["max_length"]
datapaths = config["datapaths"]
embed_config = config["embeddings"]
train_config = config["train"]
optim_config = config["optim"]

# Load the data
print("Loading the datasets...")
max_length = config["max_length"]
embedding_path = None
if embed_config["use_pretrained_embeddings"]:
    embedding_path = embed_config["embeddings_path"]
datasets, word2index, word2vec = \
    load_datasets(datapaths, embedding_path, max_length)

# Create the embedding matrix
print("Creating the embedding matrix...")
embeddings = \
    create_embedding_matrix(
        embed_config["embeddings_size"], 
        word2index, 
        word2vec
    )

# Load the model
print("Loading the model...")
model = setup_model(embeddings, config)

# Create the loss function
loss_fn = nn.BCEWithLogitsLoss()

# Create the optimizer
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=optim_config["lr"],
    weight_decay=optim_config["weight_decay"])

# Create metrics dict
metrics = {}    # TODO: Implement custom metrics functions to track

# Create history dict
history = defaultdict(list)

# Train the model
print("Training the model...")
trainset = datasets["train"]
valset = datasets["val"]
model = train(model, loss_fn, optimizer, metrics, history, trainset, valset, 
            train_config)
