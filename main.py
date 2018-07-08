# coding=utf-8

from setup import read_config
from setup import setup_model
from train import train

# Create the model
config = read_config("config.json")
model, index2word, word2index = setup_model(config)

# Load the data
# TODO: Write functions to read in the training and validation datasets

# Train the model
model = train(model)

