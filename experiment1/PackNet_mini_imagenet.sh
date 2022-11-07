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
one_shot_prune_perc=0.6
arch='vgg16_bn_mini_imagenet'
finetune_epochs=100
prune_epochs=30


for task_id in `seq 1 10`; do

  # Finetune tasks
  if [ "$task_id" != "1" ]
  then
      CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_mini_imagenet_main.py \
          --arch $arch \
          --dataset ${dataset[task_id]} --num_classes 10 \
          --lr 1e-2 \
          --weight_decay 4e-5 \
          --save_folder checkpoints/PackNet/experiment1/$arch/${dataset[task_id]}/scratch \
          --load_folder checkpoints/PackNet/experiment1/$arch/${dataset[task_id-1]}/one_shot_prune \
          --epochs $finetune_epochs \
          --mode finetune
  else
      CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_mini_imagenet_main.py \
          --arch $arch \
          --dataset ${dataset[task_id]} --num_classes 10 \
          --lr 1e-2 \
          --weight_decay 4e-5 \
          --save_folder checkpoints/PackNet/experiment1/$arch/${dataset[task_id]}/scratch \
          --epochs $finetune_epochs \
          --mode finetune
  fi

  # Prune tasks
  CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_mini_imagenet_main.py \
      --arch $arch \
      --dataset ${dataset[task_id]} --num_classes 10 \
      --lr 1e-3 \
      --weight_decay 4e-5 \
      --save_folder checkpoints/PackNet/experiment1/$arch/${dataset[task_id]}/one_shot_prune \
      --load_folder checkpoints/PackNet/experiment1/$arch/${dataset[task_id]}/scratch \
      --epochs $prune_epochs \
      --mode prune \
      --one_shot_prune_perc $one_shot_prune_perc
done


# Evaluate tasks
for history_id in `seq 1 10`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_mini_imagenet_main.py \
        --arch $arch \
        --dataset ${dataset[history_id]} --num_classes 10 \
        --load_folder checkpoints/PackNet/experiment1/$arch/${dataset[10]}/one_shot_prune \
        --mode inference
done
