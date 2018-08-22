# TODO: Implement unittests for setup.py

from setup import read_config
from setup import setup_model

config = read_config("config.json")
model, index2word, word2index = setup_model(datasets, config)