# coding=utf-8

import argparse
import os
import pandas as pd
import torch
import torch.nn as nn

from model.attention.utils import compute_attention_matrix
from model.attention.utils import cosine 
from model.attention.utils import euclidean
from model.attention.utils import manhattan
from model.pooling.allap import AllAP
from setup import read_config
from setup import setup_dataset
from setup import setup_model
from setup import setup_word_vectors
from utils import load_checkpoint
from vis_utils import make_block
from vis_utils import plot_attention_matrix

# Move to GPU if available
USE_CUDA = torch.cuda.is_available()

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="path to the checkpoint file")
parser.add_argument("--examples_path", type=str, help="path to examples file")
parser.add_argument("--output_dir", type=str, help="output path for generated plots")
args = parser.parse_args()

# Sanity check inputs
assert(os.path.isfile(args.config_path))
assert(os.path.isfile(args.checkpoint_path))
assert(os.path.isfile(args.examples_path))

# Read in the configuration file
print("Reading config file from: {}".format(args.config_path))
config = read_config(args.config_path)

# Create the model and overwrite its weights
print("Creating model with weights from: {}".format(args.checkpoint_path))
model = setup_model(config)
state = load_checkpoint(args.checkpoint_path)
model_dict, _, _, _ = state
model.load_state_dict(model_dict)

# Create the individual blocks of the model
use_all_layers = config["model"]["use_all_layers"]
max_length = config["model"]["max_length"]
embeddings_size = config["embeddings"]["size"]
block_configs = config["model"]["blocks"]
blocks = []
output_sizes = [embeddings_size]
for i, block_config in enumerate(block_configs):
    block, output_size = make_block(model_dict, "block.{}.".format(i), max_length, block_config)
    block = block.cuda() if USE_CUDA else block
    block = block.eval()
    blocks.append(block)
    output_sizes.append(output_size)

# Create the final fully connected layer
final_size = sum(output_sizes) if use_all_layers else output_sizes[-1]
fc = nn.Linear(2 * final_size, 2) # get logits
fc = fc.cuda() if USE_CUDA else fc
fc_state_dict = fc.state_dict()
fc_state_dict["weight"] = model_dict.get("fc.weight") 
fc_state_dict["bias"] = model_dict.get("fc.bias")
fc.load_state_dict(fc_state_dict)

# Create other layers needed for prediction
all_ap = AllAP() # to compute all-ap outputs
all_ap = all_ap.cuda() if USE_CUDA else all_ap
softmax = nn.Softmax(dim=0) # get class probabilities
softmax = softmax.cuda() if USE_CUDA else all_ap

# Load in the example dataset
print("Loading examples from: {}".format(args.examples_path))
examples = pd.read_csv(args.examples_path)
print(examples.head())
word_vectors = setup_word_vectors(config)
example_dataset, examples = setup_dataset(examples, word_vectors, embeddings_size, max_length)
features, label = example_dataset[0]
print(any(torch.isnan(features).tolist()))
print(example_dataset[:5])
print(examples[:5])

# Save the predictiosn for each example
pred_file = os.path.join(args.output_dir, "predictions.csv")
with open(pred_file, "w") as f:

    # Create heat maps for the examples
    for i, (example, (features, label)) in enumerate(zip(examples, example_dataset)):

        # Store all-ap outputs for final prediction
        outputs0 = []
        outputs1 = []

        # Move to GPU
        features = features.cuda() if USE_CUDA else features
        label = label.cuda() if USE_CUDA else label

        # Sanity check
        print(features.shape)
        print(label.shape)
        assert(not any(torch.isnan(features).tolist()))
        assert(not any(torch.isnan(label).tolist()))

        # Create directory to store plots
        prefix = "example{}".format(i)
        plot_dir = os.path.join(args.output_dir, prefix)
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        
        # Get features for each question
        x0 = features[0].view(1, 1, max_length, embeddings_size)
        x1 = features[1].view(1, 1, max_length, embeddings_size)

        # Store all-ap outputs for input layer
        outputs0.append(all_ap(x0))
        outputs1.append(all_ap(x1))

        # Generate initial attention distribution
        A = compute_attention_matrix(x0, x1, manhattan)
        A = A.squeeze().cpu().numpy()
        filename = "{}_input_attn.png".format(prefix)
        filepath = os.path.join(plot_dir, filename)
        plot_attention_matrix(A, example[0], example[1], filepath)

        # Generate attention distribution for blocks
        for j, block in enumerate(blocks):
            
            # Get outputs for next block
            x0, x1 = x0.detach(), x1.detach()
            x0, x1, a0, a1 = block(x0, x1)

            # Sanity check
            assert(not any(torch.isnan(x0).tolist()))
            assert(not any(torch.isnan(x1).tolist()))

            # Store all-ap outputs for this block
            outputs0.append(a0)
            outputs1.append(a1)

            # Generate attention distribution
            A = compute_attention_matrix(x0, x1, manhattan)
            A = A.squeeze().detach().cpu().numpy()
            filename = "{}_block{}_attn.png".format(prefix, j)
            filepath = os.path.join(plot_dir, filename)
            plot_attention_matrix(A, example[0], example[1], filepath)

        # Predict final output
        if use_all_layers:
            outputs = torch.cat(outputs0 + outputs1, dim=0)
        else:
            outputs = torch.cat([outputs0[-1], outputs1[-1]], dim=0)
        outputs = outputs.cuda() if USE_CUDA else outputs
        assert(not any(torch.isnan(x0).tolist()))
        logits = fc(outputs)
        probs = softmax(logits).tolist()
        print(probs)

        # Write the text with the result
        f.write("{},{},{}\n".format(example[0], example[1], probs))


