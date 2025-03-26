export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0

export NCCL_TIMEOUT=3600  # 1 hour
export TORCH_NCCL_TRACE_BUFFER_SIZE=33554432
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

export WANDB_API_KEY="638aa591e9881cd840eb171df3f625bcd7613d14"



LLM_VERSION="Qwen/Qwen2-7B-Instruct"
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

BASE_RUN_NAME="llavanext"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Stage 2
PROMPT_VERSION="qwen_1_5"
# RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-correct_memory_adapter"
RUN_NAME="llava-onevision-correct_memory_adapter"
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-ov" # replace it with your last checkpoint training from single image collection
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

NUM_GPUS=4
NNODES=$SLURM_NNODES
RANK=$SLURM_PROCID

MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_ADDR=$(getent hosts $MASTER_NODE | awk '{print $1}')
export MASTER_PORT=$(shuf -i 49152-65535 -n 1)  # IANA动态端口范围


echo "[RANK $RANK] MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"


srun --mpi=pmi2 --cpu-bind=none \
torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path /hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/LLaVA-NeXT/scripts/train/video_datasets.yaml \
    --image_folder /hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/llava-video/videos \
    --video_folder /hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/llava-video/videos \
    --mm_tunable_parts="attention_model,mm_mlp_adapter" \
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
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir /hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/checkpoints/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --force_sample True \
    --frames_upbound 128   # 32 initially
exit 0;

# You can delete the sdpa attn_implementation if you want to use flash attn
