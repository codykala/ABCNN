#!/bin/bash

python visualization.py \
    --config_path config.json \
    --checkpoint_path checkpoints/quora/word2vec/google_news/best_checkpoint \
    --examples_path data/hulo/hulo_demo_examples.csv \
    --output_dir plots/hulo/word2vec/google_news/hulo_demo_examples
