# coding=utf-8

import pandas as pd
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

from data.dataset import QuestionDataset
from setup import read_config
from setup import setup_model
from train import train

# Create the model
config = read_config("config.json")
model, index2word, word2index = setup_model(config)

# Create the loss function
loss_fn = nn.BCEWithLogitsLoss()

# Create the optimizer
optimizer = optim.Adam(model.parameters())

# Create metrics dict
metrics = {}    # TODO: Implement custom metrics functions to track

# Create history dict
history = defaultdict(list)

# Load the data
trainset = QuestionDataset("data/quora/train.csv", word2index)
valset = QuestionDataset("data/quora/val.csv", word2index)

# Train the model
model = train(model, loss_fn, optimizer, metrics, history, trainset, valset, config["train"])

