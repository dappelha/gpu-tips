#!/bin/bash

world_rank=$PMIX_RANK
let local_size=$RANKS_PER_SOCKET
let local_rank=$(expr $world_rank % $local_size)
#let num_devices_socket=$(nvidia-smi -L | grep -c GPU)
if [ -z $RANKS_PER_GPU ]; then
    echo "you must export RANKS_PER_GPU...exiting"
    exit
fi

let device=$local_rank/$RANKS_PER_GPU
    
#echo "nvidia-smi -L = $(nvidia-smi -L)"
echo "local_size = $local_size"
echo "local_rank = $local_rank, device = $device"

export CUDA_VISIBLE_DEVICES=$device

export CUDA_CACHE_PATH=/dev/shm/$USER/nvcache_$PMIX_RANK

executable=$1

shift

$executable "$@"

rm -rf /dev/shm/$USER/nvcache_$PMIX_RANK
