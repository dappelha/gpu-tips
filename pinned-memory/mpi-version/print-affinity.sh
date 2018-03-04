#!/bin/bash
echo " ---> Affinity `hostname`: $PMIX_RANK  `taskset -pc $$` $(nvidia-smi --query-gpu=index,uuid --format=csv,noheader -i $CUDA_VISIBLE_DEVICES)" #- CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES"
