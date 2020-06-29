#!/bin/bash

#gpu_num=$1
#image_A=$2
#image_B=$3
target_layer=$1
loss_type=$2
data_type=$3

image_name='n02381460_78.jpg'

CUDA_VISIBLE_DEVICES=1 python3 train2_act.py --dataroot ./datasets/apple2orange/apple2orange --name apple_cyclegan --model cycle_gan --gpu_ids -1 --image_A 'n07740461_3433.jpg' --image_B 'n07749192_78.jpg' --target_layer $target_layer --loss_type $loss_type --data_type $data_type --threshold 0.1
CUDA_VISIBLE_DEVICES=1 python3 train2_act.py --dataroot ./datasets/apple2orange/apple2orange --name apple_cyclegan --model cycle_gan --gpu_ids -1 --image_A 'n07740461_3433.jpg' --image_B 'n07749192_78.jpg' --target_layer $target_layer --loss_type $loss_type --data_type $data_type --threshold 0.2
CUDA_VISIBLE_DEVICES=1 python3 train2_act.py --dataroot ./datasets/apple2orange/apple2orange --name apple_cyclegan --model cycle_gan --gpu_ids -1 --image_A 'n07740461_3433.jpg' --image_B 'n07749192_78.jpg' --target_layer $target_layer --loss_type $loss_type --data_type $data_type --threshold 0.3
CUDA_VISIBLE_DEVICES=1 python3 train2_act.py --dataroot ./datasets/apple2orange/apple2orange --name apple_cyclegan --model cycle_gan --gpu_ids -1 --image_A 'n07740461_3433.jpg' --image_B 'n07749192_78.jpg' --target_layer $target_layer --loss_type $loss_type --data_type $data_type --threshold 0.4
CUDA_VISIBLE_DEVICES=1 python3 train2_act.py --dataroot ./datasets/apple2orange/apple2orange --name apple_cyclegan --model cycle_gan --gpu_ids -1 --image_A 'n07740461_3433.jpg' --image_B 'n07749192_78.jpg' --target_layer $target_layer --loss_type $loss_type --data_type $data_type --threshold 0.5
CUDA_VISIBLE_DEVICES=1 python3 train2_act.py --dataroot ./datasets/apple2orange/apple2orange --name apple_cyclegan --model cycle_gan --gpu_ids -1 --image_A 'n07740461_3433.jpg' --image_B 'n07749192_78.jpg' --target_layer $target_layer --loss_type $loss_type --data_type $data_type --threshold 0.6
CUDA_VISIBLE_DEVICES=1 python3 train2_act.py --dataroot ./datasets/apple2orange/apple2orange --name apple_cyclegan --model cycle_gan --gpu_ids -1 --image_A 'n07740461_3433.jpg' --image_B 'n07749192_78.jpg' --target_layer $target_layer --loss_type $loss_type --data_type $data_type --threshold 0.7
CUDA_VISIBLE_DEVICES=1 python3 train2_act.py --dataroot ./datasets/apple2orange/apple2orange --name apple_cyclegan --model cycle_gan --gpu_ids -1 --image_A 'n07740461_3433.jpg' --image_B 'n07749192_78.jpg' --target_layer $target_layer --loss_type $loss_type --data_type $data_type --threshold 0.8
CUDA_VISIBLE_DEVICES=1 python3 train2_act.py --dataroot ./datasets/apple2orange/apple2orange --name apple_cyclegan --model cycle_gan --gpu_ids -1 --image_A 'n07740461_3433.jpg' --image_B 'n07749192_78.jpg' --target_layer $target_layer --loss_type $loss_type --data_type $data_type --threshold 0.9
