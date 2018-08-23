#!/bin/bash
#
# =============================================================================
#
# A convenience script for training the ABCNN model. For more information on
# each of the command line arguments, use the following command:
#
#	python main.py --help
#
# =============================================================================

python src/main.py \
	--train \
	--config /home/cody/abcnn/configs/quora/abcnn3.yaml
