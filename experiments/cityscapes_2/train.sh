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
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29785 ../../train.py --config=config.yaml --seed ${seed}  2>&1 | tee log_${seed}_$now.txt
mkdir -p checkpoints/result 
python ../../eval.py --base_size=2048 --scales 1.0 --config=config.yaml --model_path=checkpoints/best_${seed}.pth --save_folder=checkpoints/result/ 2>&1 | tee checkpoints/result/eva-${seed}.log
python ../../eval.py --base_size=2048 --scales 1.0 --config=config.yaml --model_path=checkpoints/epoch_99_${seed}.pth --save_folder=checkpoints/result/ 2>&1 | tee checkpoints/result/eva_99-${seed}.log
done