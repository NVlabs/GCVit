#!/bin/bash
NUM_PROC=1
DATA_PATH="./ImageNet/ImageNet2012"
DATA_PATH="/home/ali/Desktop/data_local/ImageNet/imagenet2012/ImageNet2012/ImageNet2012"
checkpoint=/home/ali/Downloads/tmp/gcvit_xxtiny_best_1k.pth.tar
BS=32

python validate.py --model gc_vit_small --checkpoint=$checkpoint --data_dir=$DATA_PATH --batch-size $BS
