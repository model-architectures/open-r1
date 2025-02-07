MODEL=../X0/ckpt/qwen2.5-7b_sft_limo/global_step_12
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=8192,gpu_memory_utilisation=0.8"
OUTPUT_DIR=data/evals/$MODEL


# AIME 2024
#TASK=aime24
#lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#    --custom-tasks src/open_r1/evaluate.py \
#    --use-chat-template \
#    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

