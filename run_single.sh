#!/bin/bash
#
# =============================================================================
#
# A convenience script for training and or evaluating the ABCNN model. For more 
# information on each of the command line arguments, use the following command:
#
#	python main.py --help
#
# =============================================================================

python src/main.py \
    /home/cody/abcnn/configs/moveworks/abcnn3.yaml \
    --path /home/cody/abcnn/checkpoints/moveworks/fasttext/tickets/abcnn3_exp/best_checkpoint \
    --log_file /home/cody/abcnn/checkpoints/moveworks/fasttext/tickets/abcnn3_exp/log_file.txt \
    --eval
