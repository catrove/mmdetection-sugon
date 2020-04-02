#!/usr/bin/env bash

set -x
JOB_NAME=$1
CONFIG=$2
WORK_DIR=work_dirs/dist_centripetalnet_mask_hg104
GPUS=16
GPUS_PER_NODE=4
CPUS_PER_TASK=4
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${PY_ARGS:-"--validate"}

srun -p caspra \
    --job-name=${JOB_NAME} \
    --gres=dcu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --work_dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}
