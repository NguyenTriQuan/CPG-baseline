#!/bin/bash
# Normally, bash shell cannot support floating point arthematic, thus, here we use `bc` package

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
setting='scratch_mul_1.5'
baseline_cifar100_acc='logs/baseline_10_mini_imagenet.txt'
max_allowed_network_width_multiplier=1.5

arch='custom_vgg_10_mini_imagenet'
finetune_epochs=100
network_width_multiplier=1
pruning_ratio_interval=0.1
lr=1e-2
lr_mask=5e-4
gradual_prune_lr=1e-3
num_classes=10
batch_size=32
total_num_tasks=10


for task_id in `seq 1 10`; do

    # Training the network on current tasks
    state=2
    while [ $state -eq 2 ]; do
        if [ "$task_id" != "1" ]
        then
            CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_mini_imagenet_main.py \
                --arch $arch \
                --dataset ${dataset[task_id]} --num_classes $num_classes \
                --lr $lr \
                --lr_mask $lr_mask \
                --batch_size $batch_size \
                --weight_decay 4e-5 \
                --save_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/scratch \
                --load_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id-1]}/gradual_prune \
                --epochs $finetune_epochs \
                --mode finetune \
                --network_width_multiplier $network_width_multiplier \
                --max_allowed_network_width_multiplier $max_allowed_network_width_multiplier \
                --baseline_acc_file $baseline_cifar100_acc \
                --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune/record.txt \
                --log_path checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/train.log \
                --total_num_tasks $total_num_tasks
        else
            CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_mini_imagenet_main.py \
                --arch $arch \
                --dataset ${dataset[task_id]} --num_classes $num_classes \
                --lr $lr \
                --lr_mask $lr_mask \
                --batch_size $batch_size \
                --weight_decay 4e-5 \
                --save_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/scratch \
                --epochs $finetune_epochs \
                --mode finetune \
                --network_width_multiplier $network_width_multiplier \
                --max_allowed_network_width_multiplier $max_allowed_network_width_multiplier \
                --baseline_acc_file $baseline_cifar100_acc \
                --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune/record.txt \
                --log_path checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/train.log \
                --total_num_tasks $total_num_tasks
        fi

        state=$?
        if [ $state -eq 2 ]
        then
            network_width_multiplier=$(bc <<< $network_width_multiplier+0.5)
            echo "New network_width_multiplier: $network_width_multiplier"
            continue
        elif [ $state -eq 3 ]
        then
            echo "You should provide the baseline_cifar100_acc.txt as criterion to decide whether the capacity of network is enough for new task"
            exit 0
        fi
    done
    nrof_epoch=0
    nrof_epoch_for_each_prune=20
    start_sparsity=0.0
    end_sparsity=0.1
    nrof_epoch=$nrof_epoch_for_each_prune

    # Prune the model after training
    if [ $state -ne 5 ]
    then
        echo $state
        # gradually pruning
        CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_mini_imagenet_main.py \
            --arch $arch \
            --dataset ${dataset[task_id]} --num_classes $num_classes \
            --lr $gradual_prune_lr \
            --lr_mask 0.0 \
            --batch_size $batch_size \
            --weight_decay 4e-5 \
            --save_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune \
            --load_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/scratch \
            --epochs $nrof_epoch \
            --mode prune \
            --initial_sparsity=$start_sparsity \
            --target_sparsity=$end_sparsity \
            --pruning_frequency=10 \
            --pruning_interval=4 \
            --baseline_acc_file $baseline_cifar100_acc \
            --network_width_multiplier $network_width_multiplier \
            --max_allowed_network_width_multiplier $max_allowed_network_width_multiplier \
            --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune/record.txt \
            --log_path checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/train.log \
            --total_num_tasks $total_num_tasks

        if [ $? -ne 6 ]
        then
            for RUN_ID in `seq 1 9`; do
                nrof_epoch=$nrof_epoch_for_each_prune
                start_sparsity=$end_sparsity
                if [ $RUN_ID -lt 9 ]
                then
                    end_sparsity=$(bc <<< $end_sparsity+$pruning_ratio_interval)
                else
                    end_sparsity=$(bc <<< $end_sparsity+0.05)
                fi

                CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_mini_imagenet_main.py \
                    --arch $arch \
                    --dataset ${dataset[task_id]} --num_classes $num_classes \
                    --lr $gradual_prune_lr \
                    --lr_mask 0.0 \
                    --batch_size $batch_size \
                    --weight_decay 4e-5 \
                    --save_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune \
                    --load_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune \
                    --epochs $nrof_epoch \
                    --mode prune \
                    --initial_sparsity=$start_sparsity \
                    --target_sparsity=$end_sparsity \
                    --pruning_frequency=10 \
                    --pruning_interval=4 \
                    --baseline_acc_file $baseline_cifar100_acc \
                    --network_width_multiplier $network_width_multiplier \
                    --max_allowed_network_width_multiplier $max_allowed_network_width_multiplier \
                    --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune/record.txt \
                    --log_path checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/train.log \
                    --total_num_tasks $total_num_tasks

                if [ $? -eq 6 ]
                then
                    break
                fi
            done
        fi
    fi

    # Choose the checkpoint that we want
    python tools/choose_appropriate_pruning_ratio_for_next_task.py \
        --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune/record.txt \
        --baseline_acc_file $baseline_cifar100_acc \
        --allow_acc_loss 0.0 \
        --dataset ${dataset[task_id]} \
        --max_allowed_network_width_multiplier $max_allowed_network_width_multiplier \
        --network_width_multiplier $network_width_multiplier \
        --log_path checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/train.log

    if [ $task_id != 1 ] && [ $state -ne 5 ]
    then
    	# Retrain piggymask and weight
    	CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_mini_imagenet_main.py \
    	    --arch $arch \
    	    --dataset ${dataset[task_id]} --num_classes $num_classes \
    	    --lr $gradual_prune_lr \
    	    --lr_mask 1e-4 \
    	    --batch_size $batch_size \
    	    --weight_decay 4e-5 \
    	    --save_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/retrain \
    	    --load_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune \
    	    --epochs 30 \
    	    --mode finetune \
    	    --network_width_multiplier $network_width_multiplier \
    	    --max_allowed_network_width_multiplier $max_allowed_network_width_multiplier \
    	    --baseline_acc_file $baseline_cifar100_acc \
    	    --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/retrain/record.txt \
    	    --log_path checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/train.log \
    	    --total_num_tasks $total_num_tasks \
    	    --finetune_again

        # If there is any improve from retraining, use that checkpoint
        python tools/choose_retrain_or_not.py \
            --save_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune \
            --load_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/retrain
    fi
done
