# coding=utf-8

import argparse
import torch
import operator
import functools

from setup import read_config
from setup import setup
from utils import freeze_weights

# Read in command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("config_file", type=str, 
    help="The path to the config file describing the model.")
parser.add_argument("--freeze", action="store_true", default=False, 
    help="Specifies whether the CNN layers are frozen.")
args = parser.parse_args()

# Setup the model
config = read_config(args.config_file)
_, model = setup(config)
if args.freeze:
    print("Freezing weights of CNN layers.")
    model = freeze_weights(model)

# Calculate total number of parameters and number of learnable parmaeters
num_learnable_params = 0
num_params = 0
for name, weight in model.named_parameters():
    params_in_weight = functools.reduce(operator.mul, list(weight.size()), 1)
    if weight.requires_grad:
        num_learnable_params += params_in_weight
    num_params += params_in_weight
    
    if weight.requires_grad:
        print("{} has: {} learnable parameters".format(name, params_in_weight))
    else:
        print("{} has: {} parameters".format(name, params_in_weight))

print("Entire model has {} parameters, of which {} parameters are learnable".format(num_params, num_learnable_params))
