#!/bin/bash

echo "=== HyperKGR + Sample ==="

python -W ignore train.py --data_path data/WN18RR --epoch 50 --train --topk 500 --layers 3 \
    2>&1 | tee results_ablation/sample_wqr_WN18RR.log

python -W ignore train.py --data_path data/nell --epoch 50 --train --topk 500 --layers 3 \
    2>&1 | tee results_ablation/sample_wqr_nell.log

python -W ignore train.py --data_path data/fb15k-237 --epoch 50 --train --topk 500 --layers 3 \
    2>&1 | tee results_ablation/sample_wqr_fb15k-237.log