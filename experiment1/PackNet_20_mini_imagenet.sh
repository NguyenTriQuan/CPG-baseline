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
    '10'
    '11'
    '12'
    '13'
    '14'
    '15'
    '16'
    '17'
    '18'
    '19'
)

GPU_ID=0
one_shot_prune_perc=0.6
arch='vgg16_bn_20_mini_imagenet'
finetune_epochs=100
prune_epochs=30
num_classes=5

for task_id in `seq 1 20`; do

  # Finetune tasks
  if [ "$task_id" != "1" ]
  then
      CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_mini_imagenet_main.py \
          --arch $arch \
          --dataset ${dataset[task_id]} --num_classes $num_classes \
          --lr 1e-2 \
          --weight_decay 4e-5 \
          --save_folder checkpoints/PackNet/experiment1/$arch/${dataset[task_id]}/scratch \
          --load_folder checkpoints/PackNet/experiment1/$arch/${dataset[task_id-1]}/one_shot_prune \
          --epochs $finetune_epochs \
          --mode finetune
  else
      CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_mini_imagenet_main.py \
          --arch $arch \
          --dataset ${dataset[task_id]} --num_classes $num_classes \
          --lr 1e-2 \
          --weight_decay 4e-5 \
          --save_folder checkpoints/PackNet/experiment1/$arch/${dataset[task_id]}/scratch \
          --epochs $finetune_epochs \
          --mode finetune
  fi

  # Prune tasks
  CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_mini_imagenet_main.py \
      --arch $arch \
      --dataset ${dataset[task_id]} --num_classes $num_classes \
      --lr 1e-3 \
      --weight_decay 4e-5 \
      --save_folder checkpoints/PackNet/experiment1/$arch/${dataset[task_id]}/one_shot_prune \
      --load_folder checkpoints/PackNet/experiment1/$arch/${dataset[task_id]}/scratch \
      --epochs $prune_epochs \
      --mode prune \
      --one_shot_prune_perc $one_shot_prune_perc
done


# Evaluate tasks
for history_id in `seq 1 20`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_mini_imagenet_main.py \
        --arch $arch \
        --dataset ${dataset[history_id]} --num_classes $num_classes \
        --load_folder checkpoints/PackNet/experiment1/$arch/${dataset[20]}/one_shot_prune \
        --mode inference
done
