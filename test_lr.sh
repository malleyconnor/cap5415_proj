#!/bin/bash
lr_to_test=("0.1" "0.05" "0.02" "0.01" "0.005" "0.002" "0.001")

for val in ${lr_to_test[@]}; do
    logdir="results/model_lr${val}"
    python im_reconstruct.py --lr ${val} --log_dir ${logdir} --num_epochs 16 
done