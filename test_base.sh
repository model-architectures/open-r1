NUM_GPUS=4

# Get the passed parameter

# Construct the model path using the provided model name
MODEL="Qwen/Qwen2.5-32B-Instruct"
# Set the model arguments
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.96"
# Define the output directory
OUTPUT_DIR="qwen_32B_results"

# AIME 2024
#TASK=aime24
# Run lighteval for the AIME 2024 task (currently commented out)
#lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#    --custom-tasks src/open_r1/evaluate.py \
#    --use-chat-template \
#    --output-dir $OUTPUT_DIR

# MATH - 500
TASK=math_500
# Run lighteval for the MATH-500 task
VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES="4,5,6,7" lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --system-prompt="Please reason step by step, and put your final answer within \boxed{}." \
    --use-chat-template \
    --output-dir $OUTPUT_DIR