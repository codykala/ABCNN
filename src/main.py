# coding=utf-8

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import TensorDataset

from setup import read_config
from setup import setup
from train import train
from train import evaluate
from utils import load_checkpoint
from utils import freeze_weights

# Use GPU if available
USE_CUDA = torch.cuda.is_available()

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="path to the config file")
parser.add_argument("--train", action="store_true", default=False, help="train a model")
parser.add_argument("--eval", action="store_true", default=False, help="evaluate a model")
parser.add_argument("--path", default=None, help="path to model checkpoint")
parser.add_argument("--freeze", action="store_true", default=False, help="freeze the weights in the conv-pool layers.")
parser.add_argument("--log_file", default=None, help="path to log file")
args = parser.parse_args()

# Sanity check command line arguments
assert(args.train or args.eval)

# Basic setup
config = read_config(args.config)
features, labels, model = setup(config)
datasets = {
    name: TensorDataset(features[name], labels[name]) 
    for name in features
}
loss_fn = nn.CrossEntropyLoss()
history = defaultdict(list)

# Load trained model if specified
if args.path is not None:
    print("Loading model from checkpoint...")
    state = load_checkpoint(args.path)
    pretrained_model_dict, _, history, _ = state
    pretrained_model_dict = {
        k: v for k, v in pretrained_model_dict.items() 
        if k != "embeddings.weight"
    }
    model_dict = model.state_dict()
    model_dict.update(pretrained_model_dict)
    model.load_state_dict(model_dict)
    print("Success!")

if args.freeze:
    print("Freezing weights of pre-trained model...")
    model = freeze_weights(model)

if args.train:
    print("Training the model...")
    trainset = datasets[config["train_set"]]
    valset = datasets[config["val_set"]]
    train_config = config["train"]
    train(model, loss_fn, history, trainset, valset, train_config)

if args.eval:
    print("Evaluating the model...")
    testset = datasets[config["test_set"]]
    batch_size = config["train"]["batch_size"]
    num_workers = config["train"]["num_workers"]
    evaluate(model, testset, loss_fn, batch_size, num_workers, log_file=args.log_file)
    
