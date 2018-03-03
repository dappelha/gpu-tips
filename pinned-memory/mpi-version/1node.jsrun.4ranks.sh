#!/bin/bash
nodes=1
gpn=4  #gpu per node
ppn=4 # processes per node
let tpr=$ppn/$gpn
echo "tasks per rs = $tpr"
let nrs=$nodes*$gpn
let nmpi=$nodes*$ppn
#--------------------------------------
cat >batch.job <<EOF
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -nnodes ${nodes}
#BSUB -q excl_${gpn}gpus
#BSUB -W 30
#---------------------------------------

ulimit -s 10240
export BIND_THREADS=yes
export USE_GOMP=yes


##export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0

echo 'starting jsrun'

jsrun --rs_per_host ${gpn} --gpu_per_rs 1 --tasks_per_rs ${tpr} --cpu_per_rs 10  --nrs ${nrs} -b packed:10 -d plane:${tpr} -D CUDA_VISIBLE_DEVICES /shared/lsf-csm/csm_cn/helper_2mps.sh ./xlftest

EOF
#---------------------------------------
bsub -core_isolation y -alloc_flags "gpumps2" <batch.job
