# coding=utf-8

import torch
import operator
import functools

from setup import read_config
from setup import setup_model

config = read_config("config.json")
model = setup_model(config)
num_params = 0
for name, weight in model.named_parameters():
    params_in_weight = functools.reduce(operator.mul, list(weight.size()), 1)
    print("{} has: {} parameters".format(name, params_in_weight))
    num_params += params_in_weight
print("Entire model has: {} parameters".format(num_params))


