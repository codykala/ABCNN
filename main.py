# coding=utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

from setup import read_config
from setup import setup_datasets_and_model
from train import train

# Get configurations
config = read_config("config.json")

# Load the model and datasets
print("Loading the datasets and model...")
datasets, model = setup_datasets_and_model(config)

# Create the loss function
loss_fn = nn.CrossEntropyLoss()

# Create the optimizer
lr = config["optim"]["lr"]
weight_decay = config["optim"]["weight_decay"]
optimizer = \
    optim.Adagrad(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

# Create history dict
history = defaultdict(list)

# Train the model
print("Training the model...")
trainset = datasets["train"]
valset = datasets["val"]
train_config = config["train"]
model = train(model, loss_fn, optimizer, history, trainset, valset, train_config)
