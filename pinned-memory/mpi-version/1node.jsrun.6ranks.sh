#!/bin/bash
# There are 21 (of 22) cores available to the application per socket
# Each core can go up to smt4
nodes=1
gpus_per_socket=3 # number of gpus to use per socket
ranks_per_gpu=1 # ranks per gpu. If greater than 1, should use mps.
let ranks_per_socket=$gpus_per_socket*$ranks_per_gpu # needs to be evenly divisible by gpus_per_socket. 
let cores_per_rank=21/$ranks_per_socket # 21 avail cores divided into the ranks.
let nmpi=2*$ranks_per_socket*$nodes  # total number of mpi ranks
let cores_per_socket=$cores_per_rank*$ranks_per_socket # this is used cores per socket (not necessarily 21)
let num_sockets=$nodes*2 #nmpi/ranks_per_socket # total number of sockets
let threads_per_rank=4*$cores_per_rank

echo "nodes = $nodes"
echo "gpus used per socket = $gpus_per_socket"
echo "ranks_per_socket = $ranks_per_socket"
echo "cores_per_rank = $cores_per_rank"
echo "used cores per socket = $cores_per_socket"

let transfersize=2*1024
#--------------------------------------
cat >batch.job <<EOF
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -nnodes ${nodes}
#BSUB -alloc_flags gpumps
#BSUB -alloc_flags smt4
#BSUB -P VEN201
#BSUB -q batch
#BSUB -W 5
#---------------------------------------

ulimit -s 10240

export OMP_NUM_THREADS=$threads_per_rank
#export CUDA_LAUNCH_BLOCKING=0

echo 'starting jsrun with'
echo "nodes = $nodes"
echo "gpus used per socket = $gpus_per_socket"
echo "ranks_per_socket = $ranks_per_socket"
echo "cores_per_rank = $cores_per_rank"
echo "used cores per socket = $cores_per_socket"

export RANKS_PER_SOCKET=$ranks_per_socket
export RANKS_PER_GPU=$ranks_per_gpu

# CHECK AFFINITY:

 jsrun --smpiargs="-mxm" --nrs ${num_sockets}  --tasks_per_rs ${ranks_per_socket} --cpu_per_rs ${cores_per_socket} \
  --gpu_per_rs ${gpus_per_socket} --bind=proportional-packed:${cores_per_rank} -d plane:${ranks_per_socket}  \
  ./device-bind.sh ./print-affinity.sh 


# PGI RUN:

# needed for OpenACC to actually delete memory when requested.  
  #export PGI_ACC_MEM_MANAGE=0

  #jsrun -e prepended --smpiargs="-mxm" --nrs ${num_sockets}  --tasks_per_rs ${ranks_per_socket} --cpu_per_rs ${cores_per_socket} \
  # --gpu_per_rs ${gpus_per_socket} --bind=proportional-packed:${cores_per_rank} -d plane:${ranks_per_socket}  \
  # ./device-bind.sh ./pgitest $transfersize

# XLF RUN:

#  jsrun -e prepended --smpiargs="-mxm" --nrs ${num_sockets}  --tasks_per_rs ${ranks_per_socket} --cpu_per_rs ${cores_per_socket} \
#   --gpu_per_rs ${gpus_per_socket} --bind=proportional-packed:${cores_per_rank} -d plane:${ranks_per_socket}  \
#   ./device-bind.sh ./xlftest ${transfersize}




EOF
#---------------------------------------
# WSC cluster:
#bsub -core_isolation y -alloc_flags "gpumps2" <batch.job
# SUMMIT:
bsub <batch.job
