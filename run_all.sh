#!/bin/bash
#
# =============================================================================
#
# A convenience script for fine-tuning a pre-trained ABCNN model on moveworks
# data. For more information on each of the command line arguments, use the
# following command:
#
#   python main.py --help
#
# =============================================================================

python src/main.py \
    /home/cody/abcnn/configs/quora/abcnn3.yaml \
    --train

python src/main.py \
    /home/cody/abcnn/configs/moveworks/abcnn3.yaml \
    --path /home/cody/abcnn/checkpoints/quora/fasttext/tickets/abcnn3_new/best_checkpoint \
    --log_file /home/cody/abcnn/checkpoints/quora/fasttext/tickets/abcnn3_new/log_file.txt \
    --freeze \
    --train \
    --eval
