#!/usr/bin/env bash
####
echo "it is runing 1 st code" ### 'fd' 'ld'
python /17739334165/LXH_iStation/Project/DCCT_VideoReID_CE_Trip_iStation/Train.py \
      --istation  --batch_size 32 --dataset mars --arch DCCT \
      --model_mode 'cnn' 'transformer' 'cca' 'hta' --num_dim 512  --layer 2  \
      --method_name DCCT_HTA_dim_Ablation --changed_thing NoFD_NoLD
