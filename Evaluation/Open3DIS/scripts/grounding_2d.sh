#!/bin/bash

dataset_cfg=${1:-'configs/scannet200.yaml'}
export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=1 python3 tools/grounding_2d.py --config $dataset_cfg