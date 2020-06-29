#!/bin/bash

#gpu_num=$1
#image_A=$2
#image_B=$3
target_layer=$1
loss_type=$2
data_type=$3

CUDA_VISIBLE_DEVICES=1 python3 train2.py --dataroot ./datasets/apple2orange/apple2orange --name apple_cyclegan --model cycle_gan --gpu_ids -1 --image_A 'n07740461_2867.jpg' --image_B 'n07749192_167.jpg' --target_layer $target_layer --loss_type $loss_type --data_type $data_type

