#!/bin/bash
#
# =============================================================================
#
# A convenience script for evaluating the ABCNN model. For more information on
# each of the command line arguments, use the following command:
#
#	python main.py --help
#
# =============================================================================

python src/main.py \
	--eval \
	--config /home/cody/abcnn_configs/abcnn3_config.json 
