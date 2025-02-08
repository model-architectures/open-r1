NUM_GPUS=4
# Check if a parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

# Get the passed parameter
MODEL_NAME=$1

# Construct the model path using the provided model name
MODEL="../X0/ckpt/${MODEL_NAME}/global_step_12"
# Set the model arguments
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=36864,gpu_memory_utilisation=0.9"
# Define the output directory
OUTPUT_DIR="data/evals/$MODEL"

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
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
