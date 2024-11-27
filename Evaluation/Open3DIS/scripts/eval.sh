#!/bin/bash

export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH

export PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python3 open3dis/evaluation/eval.py
# CUDA_VISIBLE_DEVICES=0 python3 open3dis/evaluation/inst_run_replica.py
#laion2b_s39b_b160k

