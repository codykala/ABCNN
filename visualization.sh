#!/bin/bash

python visualization.py \
    --config_path config.json \
    --checkpoint_path checkpoints/quora/fasttext/wiki/checkpoint_epoch_5 \
    --examples_path data/quora/demo_examples.csv \
    --output_dir plots/quora/fasttext/wiki/demo_examples
