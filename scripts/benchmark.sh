#!/bin/bash

set -x
export CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

source "${main_dir}/configs/model_glm_130b_mlu.sh"

ARGS="${main_dir}/benchmark.py \
       --distributed-backend cncl \
       --mode inference \
       $MODEL_ARGS"

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXP_NAME=${TIMESTAMP}

mkdir -p logs

export dist_port=20233
run_cmd="python -m torch.distributed.launch --master_port $dist_port \
       --nproc_per_node $MP_SIZE ${ARGS}"
eval ${run_cmd} 2>&1 | tee logs/${EXP_NAME}.log
