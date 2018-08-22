# coding=utf-8

import torch
import operator
import functools

from setup import read_config
from setup import setup

config = read_config("/home/cody/abcnn_configs/abcnn3_config.json")
_, model = setup(config)
num_params = 0
for name, weight in model.named_parameters():
    params_in_weight = functools.reduce(operator.mul, list(weight.size()), 1)
    print("{} has: {} parameters".format(name, params_in_weight))
    num_params += params_in_weight
print("Entire model has: {} parameters".format(num_params))


