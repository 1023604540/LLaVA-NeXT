export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0

export NCCL_DEBUG=DEBUG
export USE_PYTORCH_KERNEL_CACHE=0

# export NCCL_DEBUG=INFO   # Uncomment for debugging
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TIMEOUT=3600  # 1 hour
# export TORCH_NCCL_TRACE_BUFFER_SIZE=33554432  # Uncomment for debugging

# The next line is very important! Solves the WatchDog TimeOut Issue
export NCCL_P2P_DISABLE=1
export CUTLASS_PATH=/hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/cutlass
export WANDB_API_KEY="638aa591e9881cd840eb171df3f625bcd7613d14"

LLM_VERSION="Qwen/Qwen2-0.5B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

BASE_RUN_NAME="llavanext"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="llava-onevision-0.5b-qwen2_KIT_position_8tokens_adapter"
# PREV_STAGE_CHECKPOINT="/hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/checkpoints/llava-onevision-qwen2-0.5b-ov" # replace it with your last checkpoint training from single image collection
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-0.5b-ov"

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

NUM_GPUS=4
NNODES=$SLURM_NNODES
#RANK=$SLURM_PROCID
RANK=$SLURM_NODEID

MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_ADDR=$(getent hosts $MASTER_NODE | awk '{print $1}')
export MASTER_PORT=$(shuf -i 49152-65535 -n 1)  # IANA动态端口范围

echo "[RANK $RANK] MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

srun --mpi=pmix --export=ALL,ACCELERATE_CPU_AFFINITY=0 \
  torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path /hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/LLaVA-NeXT/scripts/train/long_train.yaml \
    --image_folder /hkfs/work/workspace/scratch/tum_tyz7686-hf_storage/videos \
    --video_folder /hkfs/work/workspace/scratch/tum_tyz7686-hf_storage/videos \
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
    --output_dir /hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/checkpoints/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --memory_transformer_lr 1e-4 \
    --memory_key_value_lr 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing False \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --force_sample False \
    --frames_upbound 250 \
    --attn_implementation "flash_attention_2"
exit 0;

# You can delete the sdpa attn_implementation if you want to use flash attn
