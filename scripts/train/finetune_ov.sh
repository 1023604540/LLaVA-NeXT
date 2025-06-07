export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=DEBUG
export NCCL_DEBUG_SUBSYS=ALL

export NCCL_TIMEOUT=3600  # 1 hour
export TORCH_NCCL_TRACE_BUFFER_SIZE=33554432
export NCCL_P2P_DISABLE=1
export WANDB_API_KEY="638aa591e9881cd840eb171df3f625bcd7613d14"
export CUDA_LAUNCH_BLOCKING=1


LLM_VERSION="Qwen/Qwen2-7B-Instruct"
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="0.5b_FAU_llava_onevision_qwen2_8tokens_adapter_correct_gate"
# PREV_STAGE_CHECKPOINT="/anvme/workspace/b232dd16-LLaVA-OV/llava-onevision-qwen2-7b-ov" # replace it with your last checkpoint training from single image collection
PREV_STAGE_CHECKPOINT="/anvme/workspace/b232dd16-LLaVA-OV/llava-onevision-qwen2-0.5b-ov"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

NUM_GPUS=8
NNODES=$SLURM_NNODES
RANK=$SLURM_PROCID
ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)  # Master node
PORT=12346



ACCELERATE_CPU_AFFINITY=0 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path /home/hpc/b232dd/b232dd16/LLaVA-OV/scripts/train/sharegpt_train.yaml \
    --image_folder /anvme/workspace/b232dd21-zyr/llava-data \
    --video_folder /anvme/workspace/b232dd16-LLaVA-OV/long_videos/video_data \
    --mm_tunable_parts="larimar_model,recurrent_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --mm_newline_position one_token \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir /anvme/workspace/b232dd16-LLaVA-OV/checkpoints/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate 2e-6 \
    --memory_transformer_lr 1e-4 \
    --memory_key_value_lr 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last False \
    --force_sample True \
    --frames_upbound 300   # 32 initially
exit 0;

# You can delete the sdpa attn_implementation if you want to use flash attn
