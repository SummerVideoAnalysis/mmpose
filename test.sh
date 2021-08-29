#!/bin/bash

# python tools/test.py \
# configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_512x512.py \
# work_dirs/higherhrnet_w32_coco_512x512/epoch_300.pth \
# --out higherhrnet_penalty_results.json

# python tools/test.py \
# configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py \
# work_dirs/hrnet_w32_coco_256x192/epoch_400.pth \
# --out 2017-11-08-bos-nyr-national_020132_results.json

python tools/test.py \
configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vipnas_res50_coco_256x192.py \
work_dirs/vipnas_res50_coco_256x192/epoch_220.pth \
--out 2017-11-08-bos-nyr-national_020132_vipnas_results.json