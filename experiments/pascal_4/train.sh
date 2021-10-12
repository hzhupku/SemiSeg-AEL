#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
PARTITION=$1
JOB_NAME=$2
ROOT=../..
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$ROOT:$PYTHONPATH
SEED=(1 2 3 4 5)
for seed in ${SEED[*]}
do
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29763 ../../train.py --config=config.yaml --seed ${seed}  2>&1 | tee log_${seed}_$now.txt
done