#!/bin/bash
NUM_PROC=1
DATA_PATH="./ImageNet/ImageNet2012"
checkpoint=./tmp/gcvit_xxtiny_best_1k.pth.tar
BS=128

python validate.py --model gc_vit_xxtiny --checkpoint=$checkpoint --data_dir=$DATA_PATH --batch-size $BS
