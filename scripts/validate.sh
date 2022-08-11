#!/bin/bash
NUM_PROC=1
DATA_PATH="./ImageNet/ImageNet2012"
checkpoint=./output/train/model_best.pth.tar
BS=128

python validate.py --model gc_vit_small --checkpoint=$checkpoint --data_dir=$DATA_PATH --batch-size $BS
