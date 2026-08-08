#!/bin/bash
# Run XTTS augmentation generation (long-running, must be run from terminal)
cd /home/sheigl/code/auralis
VIRTUAL_ENV=.venv ONEAPI_DEVICE_SELECTOR="level_zero:1" nohup uv run \
  /home/sheigl/code/kokoclone-train/tools/build_xtts_aug.py \
  > /home/sheigl/code/kokoclone-train/build_xtts_aug.log 2>&1 &
echo "XTTS augmentation PID: $!"
echo "Log: /home/sheigl/code/kokoclone-train/build_xtts_aug.log"
echo "Monitor: tail -f /home/sheigl/code/kokoclone-train/build_xtts_aug.log"
