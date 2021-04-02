# scripts for runnig paired word-definitions

# base-1gpu-fp16-slow
# GPU=0 NUM_GPU=1 ARGS="--fp16 --fp16-init-scale 1" MODEL=checkpoints/roberta.base/model DATA_ROOT=data-bin DATA=bookcorpus_all ARCH=roberta_base MAX_EPOCH=10 WARMUP_UPDATES=295 MAX_SENTENCES=32 UPDATE_FREQ=64 TOKENS_PER_SAMPLE=128 MAX_POSITIONS=128

# large-4gpu-fp16-slow
# GPU=0,1,2,3 NUM_GPU=4 ARGS="--fp16 --fp16-init-scale 1" MODEL=checkpoints/roberta.large/model DATA_ROOT=data-bin DATA=bookcorpus_all ARCH=roberta_large MAX_EPOCH=10 WARMUP_UPDATES=295 MAX_SENTENCES=4 UPDATE_FREQ=128 TOKENS_PER_SAMPLE=256 MAX_POSITIONS=256 USE_LAMB=1

# pair no relation, mask target word, mask definition
# GPU=0,1 NUM_GPU=2 PORT=29500 CUSTOM_ID="norel" ARGS="--fp16 --fp16-init-scale 1 --no-relation-prediction" MODEL=checkpoints/roberta-large DATA_ROOT=data-bin DATA=bookcorpus_all ARCH=roberta_large MAX_EPOCH=10 WARMUP_UPDATES=295 MAX_SENTENCES=4 UPDATE_FREQ=256 TOKENS_PER_SAMPLE=256 MAX_POSITIONS=256

# pair no definition, could mask target word
# GPU=2,3 NUM_GPU=2 PORT=29501 CUSTOM_ID="nodef" ARGS="--fp16 --fp16-init-scale 1 --no-definition" MODEL=checkpoints/roberta-large DATA_ROOT=data-bin DATA=bookcorpus_all ARCH=roberta_large MAX_EPOCH=10 WARMUP_UPDATES=295 MAX_SENTENCES=4 UPDATE_FREQ=256 TOKENS_PER_SAMPLE=256 MAX_POSITIONS=256

# pair mask target word only
# GPU=4,5 NUM_GPU=2 PORT=29502 CUSTOM_ID="masklemma" ARGS="--fp16 --fp16-init-scale 1 --no-masking-definition" MODEL=checkpoints/roberta-large DATA_ROOT=data-bin DATA=bookcorpus_all ARCH=roberta_large MAX_EPOCH=10 WARMUP_UPDATES=295 MAX_SENTENCES=4 UPDATE_FREQ=256 TOKENS_PER_SAMPLE=256 MAX_POSITIONS=256

# pair mask definition only
# GPU=6,7 NUM_GPU=2 PORT=29503 CUSTOM_ID="maskdef" ARGS="--fp16 --fp16-init-scale 1 --no-masking-target-word" MODEL=checkpoints/roberta-large DATA_ROOT=data-bin DATA=bookcorpus_all ARCH=roberta_large MAX_EPOCH=10 WARMUP_UPDATES=295 MAX_SENTENCES=4 UPDATE_FREQ=256 TOKENS_PER_SAMPLE=256 MAX_POSITIONS=256

eval "$(conda shell.bash hook)"
conda activate sempre

# required
MODEL=${MODEL:-"checkpoints/roberta.base/model"}
DATA=${DATA:-"bookcorpus_all"}
ARCH=${ARCH:-"roberta-base"}

# hyperparameters

# each epoch for wn contains 691 batches of 2048 samples
# total updates are for 10 epochs
# warmup updates are 6%
BASE_UPDATES=${BASE_UPDATES:-691}
MAX_EPOCH=${MAX_EPOCH:-10}
LR=${LR:-0.00002}
TOKENS_PER_SAMPLE=${TOKENS_PER_SAMPLE:-256}
MAX_POSITIONS=${MAX_POSITIONS:-256}
MAX_SENTENCES=${MAX_SENTENCES:-32}
UPDATE_FREQ=${UPDATE_FREQ:-16}
CUSTOM_ID=${CUSTOM_ID:-""}

# default
DATA_ROOT=${DATA_ROOT:-"data-bin"}
if [[ ! -v GPU ]]; then
    GPU=${GPU:-0}
fi
NUM_GPU=${NUM_GPU:-1}
PORT=${PORT:-29500}


# optional
USER_DIR=${USER_DIR:-"../sempre"}
ARGS=${ARGS:-""}
SEED=${SEED:-1234}
USE_LAMB=${USE_LAMB:-0}

# inferred
if [[ ! -z "$CUSTOM_ID" ]]; then
    ID="${CUSTOM_ID}-batchsize$((MAX_SENTENCES*UPDATE_FREQ*NUM_GPU))-seed${SEED}"
else
    ID="batchsize$((MAX_SENTENCES*UPDATE_FREQ*NUM_GPU))-seed${SEED}"
fi


if [[ $MAX_EPOCH -ne 10 ]]; then
    ID="${ID}-me${MAX_EPOCH}"
fi

TOTAL_TOTAL_UPDATES=$((BASE_UPDATES*MAX_EPOCH))
TOTAL_NUM_UPDATES=$((BASE_UPDATES*MAX_EPOCH))

if [[ ! -v WARMUP_UPDATES ]]; then
    WARMUP_UPDATES=$((TOTAL_TOTAL_UPDATES/10))
fi

CHECKPOINT_ROOT="$MODEL/$ID"

if [[ ! -z "$USER_DIR" ]]; then
    ARGS="${ARGS} --user-dir ${USER_DIR} "
fi

if [[ -z "$GPU" ]]; then
    ARGS="${ARGS} --cpu "
    CMD="python train.py"
    MAX_SENTENCES=$((MAX_SENTENCES*UPDATE_FREQ))
    UPDATE_FREQ=1
else
    if [[ $NUM_GPU -gt 1 ]]; then
        CMD="python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT train.py "
        ARGS="${ARGS} --ddp-backend=no_c10d --distributed-no-spawn"
    else
        CMD="python train.py"
    fi
fi

echo "running command: $CMD"
echo "running model ID: $ID"
echo "running extra args: $ARGS"

if [[ -f "$CHECKPOINT_ROOT/checkpoint_last.pt" ]]; then
    # continue training
    if [[ $USE_LAMB -eq 0 ]]; then
        CUDA_VISIBLE_DEVICES=$GPU $CMD "$DATA_ROOT/$DATA" \
            $ARGS \
            --required-batch-size-multiple 2 \
            --task masked_def_lm --criterion masked_lm_prediction \
            --separator-token "</s>" --init-token "<s>" --mask-token "<mask>" --def-sep-token "<:>" --surround-token "<sur>" \
            --num-workers 4 --tokens-per-sample $TOKENS_PER_SAMPLE --max-positions $MAX_POSITIONS \
            --mask-prob 0.15 --leave-unmasked-prob 0.1 --random-token-prob 0.1 \
            --bpe gpt2 --arch $ARCH \
            --activation-fn gelu --pooler-activation-fn tanh \
            --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 --pooler-dropout 0.1 \
            --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 --weight-decay 0.01 \
            --lr-scheduler polynomial_decay --lr $LR \
            --max-epoch 2  --max-update $TOTAL_NUM_UPDATES \
            --total-num-update $TOTAL_TOTAL_UPDATES --warmup-updates $WARMUP_UPDATES \
            --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
            --skip-invalid-size-inputs-valid-test \
            --save-dir "$CHECKPOINT_ROOT" --seed $SEED \
            --save-interval 1 --keep-last-epochs 10 \
            --log-format simple --log-interval 5 --disable-validation |
            tee -a "$CHECKPOINT_ROOT/${ID}-train.txt"
    else
        CUDA_VISIBLE_DEVICES=$GPU $CMD "$DATA_ROOT/$DATA" \
            $ARGS \
            --required-batch-size-multiple 2 \
            --task masked_def_lm --criterion masked_lm_prediction \
            --separator-token "</s>" --init-token "<s>" --mask-token "<mask>" --def-sep-token "<:>" --surround-token "<sur>" \
            --num-workers 4 --tokens-per-sample $TOKENS_PER_SAMPLE --max-positions $MAX_POSITIONS \
            --mask-prob 0.15 --leave-unmasked-prob 0.1 --random-token-prob 0.1 \
            --bpe gpt2 --arch $ARCH \
            --activation-fn gelu --pooler-activation-fn tanh \
            --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 --pooler-dropout 0.1 \
            --optimizer lamb --lamb-betas '(0.9,0.999)' --lamb-eps 1e-6 --clip-norm 0.0 --weight-decay 0.01 \
            --lr-scheduler polynomial_decay --lr $LR \
            --max-epoch 2  --max-update $TOTAL_NUM_UPDATES \
            --total-num-update $TOTAL_TOTAL_UPDATES --warmup-updates $WARMUP_UPDATES \
            --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
            --skip-invalid-size-inputs-valid-test \
            --save-dir "$CHECKPOINT_ROOT" --seed $SEED \
            --save-interval 1 --keep-last-epochs 10 \
            --log-format simple --log-interval 5 --disable-validation |
            tee -a "$CHECKPOINT_ROOT/${ID}-train.txt"
    fi
else
    # not started, run training
    mkdir -p "$CHECKPOINT_ROOT"

    CUDA_VISIBLE_DEVICES=$GPU $CMD "$DATA_ROOT/$DATA" \
        $ARGS \
        --restore-file "${MODEL}.pt" \
        --reset-optimizer --reset-dataloader --reset-meters \
        --required-batch-size-multiple 2 \
        --task masked_def_lm --criterion masked_lm_prediction \
        --separator-token "</s>" --init-token "<s>" --mask-token "<mask>" --def-sep-token "<:>" --surround-token "<sur>" \
        --num-workers 4 --tokens-per-sample $TOKENS_PER_SAMPLE --max-positions $MAX_POSITIONS \
        --mask-prob 0.15 --leave-unmasked-prob 0.1 --random-token-prob 0.1 \
        --bpe gpt2 --arch $ARCH \
        --activation-fn gelu --pooler-activation-fn tanh \
        --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 --pooler-dropout 0.1 \
        --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 --weight-decay 0.01 \
        --lr-scheduler polynomial_decay --lr $LR \
        --max-epoch 2  --max-update $TOTAL_NUM_UPDATES \
        --total-num-update $TOTAL_TOTAL_UPDATES --warmup-updates $WARMUP_UPDATES \
        --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
        --skip-invalid-size-inputs-valid-test \
        --save-dir "$CHECKPOINT_ROOT" --seed $SEED \
        --save-interval 1 --keep-last-epochs 10 \
        --log-format simple --log-interval 5 --disable-validation |
        tee "$CHECKPOINT_ROOT/${ID}-train.txt"
fi