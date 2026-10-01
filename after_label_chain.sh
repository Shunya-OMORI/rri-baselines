#!/bin/bash
# 手元 GPU: 正解を変えた提案手法の学習(run_label_mid_linear.sh)が終わるのを待ってから, 続きを 1 本ずつ流す
set -e
cd /home/shunya/research/rri-baselines
while pgrep -f run_label_mid_linear.sh > /dev/null; do sleep 120; done
echo "label chain finished at $(date)"
(cd /home/shunya/research/mmwave-rri && .venv/bin/python ../rri-baselines/scripts/label_model_compare.py) > results/label_model_compare.log 2>&1 || true
if [ -x plain_fixed_chain.sh ]; then ./plain_fixed_chain.sh; fi
