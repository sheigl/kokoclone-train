#!/bin/bash
export PATH="/home/sheigl/code/kokoclone-train/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export ONEAPI_DEVICE_SELECTOR="level_zero:0"
export PYTHONUNBUFFERED=1
cd /home/sheigl/code/kokoclone-train || exit 1
.venv/bin/python -m kokoclone_train train \
  --name am_jefferson \
  --audio-dir training/am_jefferson \
  --output-dir voices \
  --use-decoder \
  --force \
  >> training_am_jefferson_v3.log 2>&1
echo "EXIT CODE: $?" >> training_am_jefferson_v3.log
