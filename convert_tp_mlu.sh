

source configs/model_glm_130b.sh

python tools/convert_tp.py \
    --input-folder $CHECKPOINT_PATH \
    --output-folder glm-130b-sat-mlu/ \
    --target-tp 16 
