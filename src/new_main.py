# coding=utf-8

import argparse
import os
from torch.utils.data import TensorDataset

from trainer.factories import loss_fn_factory
from trainer.factories import optimizer_factory
from trainer.factories import scheduler_factory
from trainer.multiclass_classifier_trainer import MulticlassClassifierTrainer
from trainer.utils import move_to_device
from setup import read_config
from setup import setup
from utils import abcnn_model_loader

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, 
    help="path to the config file")
parser.add_argument("trainset", type=str, 
    help="the name of the dataset (key in data_paths) to use for training.")
parser.add_argument("valset", type=str, 
    help="the name of the dataset (key in data_paths) to use for validation.")
parser.add_argument("testset", type=str, 
    help="the name of the dataset (key in data_paths) to use for testing.")
parser.add_argument("-l", "--load", type=str, default=None, 
    help="load a pre-trained model from a checkpoint file.")
parser.add_argument("-t", "--train", action="store_true", default=False, 
    help="train a model")
parser.add_argument("-p", "--predict", action="store_true", default=False, 
    help="evaluate a model")
args = parser.parse_args()

# Sanity check command line arguments
assert(os.path.isfile(args.config_path))
assert(args.load is None or os.path.isfile(args.load))
assert(args.train or args.predict)

# Initial setup
config = read_config(args.config_path)
features, labels, model = setup(config["model"])
model = move_to_device(config["trainer"]["device"], model) # model needs to be on correct device BEFORE optimizer is initialized
datasets = {
    name: TensorDataset(features[name], labels[name])
    for name in features
}
loss_fn = loss_fn_factory(config["loss_fn"])
optimizer = optimizer_factory(config["optimizer"], model.parameters())
scheduler = scheduler_factory(config["scheduler"], optimizer)
trainer = MulticlassClassifierTrainer(config["trainer"])

# Load a pre-trained model
if args.load:
    model, optimizer = abcnn_model_loader(args.load, model, optimizer)

# Train the model
if args.train:
    trainset = datasets[args.trainset]
    valset = datasets[args.valset] if args.valset else None
    trainer.train(loss_fn, model, optimizer, trainset, scheduler=scheduler, valset=valset)

# Make predictions
if args.predict:
    testset = datasets[args.testset]
    trainer.predict(testset)
