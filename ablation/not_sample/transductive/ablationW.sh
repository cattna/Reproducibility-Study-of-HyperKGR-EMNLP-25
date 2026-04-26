#!/bin/bash

echo "=== HYPERKGR Non-Sample ==="

python -W ignore train.py --data_path data/WN18RR \
    2>&1 | tee results_ablation/nosample_WN18RR.log

python -W ignore train.py --data_path data/nell \
    2>&1 | tee results_ablation/nosample_nell.log	

python -W ignore train.py --data_path data/fb15k-237 \
    2>&1 | tee results_ablation/nosample_fb15k-237.log