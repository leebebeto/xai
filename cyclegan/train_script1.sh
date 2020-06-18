#!/bin/bash

#gpu_num=$1
#image_A=$2
#image_B=$3
target_layer=$1
loss_type=$2

CUDA_VISIBLE_DEVICES=1 python3 train1.py --dataroot ./datasets/horse2zebra --name horse_cyclegan --model cycle_gan --gpu_ids -1 --image_A 'n02381460_331.jpg' --image_B 'n02391049_7851.jpg' --target_layer $target_layer --loss_type $loss_type

