#!/bin/bash
#
# =============================================================================
#
# A convenience script for training and or evaluating the ABCNN model. For more 
# information on each of the command line arguments, use the following command:
#
#	python src/main.py --help
#
# =============================================================================

python src/main.py \
    /home/cody/abcnn/ABCNN/src/config.yaml \
    moveworks_train \
    moveworks_val \
    moveworks_test \
    --train 

