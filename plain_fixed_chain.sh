#!/bin/bash
# 改訂 2 版のプレーンな網を手元 GPU で 1 本ずつ(after_label_chain.sh から呼ばれる)
set -e
cd /home/shunya/research/rri-baselines
for net in transformer wavenet tcn; do
  .venv/bin/python -m rri_baselines.plain.run train $net 0 1 2 3 4 5 6 7 8 9 10 > logs/plain2_$net.log 2>&1
done
.venv/bin/python -m rri_baselines.plain.run evaluate > logs/plain2_evaluate.log 2>&1
echo PLAIN2_DONE
