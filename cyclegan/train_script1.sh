#!/bin/bash

#gpu_num=$1
#image_A=$2
#image_B=$3
target_layer=$1 # 0, 1, 2, 3
loss_type=$2 # cycle
data_type=$3 # horse or apple


CUDA_VISIBLE_DEVICES=1 python3 train1.py --dataroot ./datasets/horse2zebra --name horse_cyclegan --model cycle_gan --gpu_ids -1 --image_A 'n02381460_69.jpg' --image_B 'n02391049_7851.jpg' --target_layer $target_layer --loss_type $loss_type --data_type $data_type

