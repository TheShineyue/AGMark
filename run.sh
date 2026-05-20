export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=None
export MODEL_NAME=qwen3
export TASK_NAME=AMBER

python3 main.py \
    --json_path None \
    --image_dir None \
    --model_path "${MODEL_PATH}" \
    --range_num 50 \
    --model_name "${MODEL_NAME}" \
    --task_name AMBER \
    --data_suffix .jpg \
    --similarity_scheme cosine \
    --max_tokens 200 \
    --min_tokens 190 
