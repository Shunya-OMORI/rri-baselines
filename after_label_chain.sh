#!/bin/bash
# 手元 GPU: 正解を変えた提案手法の学習が終わった後の続き(表 → プレーンな網の改訂 2 版)
set -e
cd /home/shunya/research/rri-baselines
(cd /home/shunya/research/mmwave-rri && .venv/bin/python ../rri-baselines/scripts/label_model_compare.py) > results/label_model_compare.log 2>&1 || true
./plain_fixed_chain.sh
