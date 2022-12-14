#!/bin/bash

dataset=(
    'None'                # dummy
    '0'
    '1'
    '2'
    '3'
    '4'
    '5'
    '6'
    '7'
    '8'
    '9'
)

GPU_ID=0
NETWORK_WIDTH_MULTIPLIER=1
ARCH='custom_vgg_10_mini_imagenet'
SETTING='scratch_mul_1.5'

for TASK_ID in `seq 1 10`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_mini_imagenet_main.py \
        --arch $ARCH \
        --dataset ${DATASETS[TASK_ID]} --num_classes 10 \
        --load_folder checkpoints/CPG/experiment1/$SETTING/$ARCH/${DATASETS[10]}/gradual_prune \
        --mode inference \
        --baseline_acc_file logs/baseline_10_mini_imagenet.txt \
        --network_width_multiplier $NETWORK_WIDTH_MULTIPLIER \
        --max_allowed_network_width_multiplier 1.5 \
        --log_path logs/10_mini_imagenet_inference.log
done
