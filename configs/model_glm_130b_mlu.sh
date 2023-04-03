MODEL_TYPE="glm-130b"
CHECKPOINT_PATH="glm-130b-sat-mlu/"
MP_SIZE=16
MODEL_ARGS="--model-parallel-size ${MP_SIZE} \
            --num-layers 70 \
            --hidden-size 12288 \
            --inner-hidden-size 32768 \
            --vocab-size 150528 \
            --num-attention-heads 96 \
            --max-sequence-length 2048 \
            --tokenizer-type icetk-glm-130B \
            --layernorm-order post \
            --load ${CHECKPOINT_PATH} \
            --skip-init \
            --fp16 "
