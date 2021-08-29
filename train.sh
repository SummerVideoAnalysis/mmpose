#!/bin/bash

python tools/train.py \
configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_512x512.py \
--cfg-options data.samples_per_gpu=2 \
# --resume-from work_dirs/hrnet_w32_coco_256x192/epoch_200.pth
