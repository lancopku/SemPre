# base-2gpu
# GPU=0,1 NUM_GPU=2 PORT=67500 MODEL=checkpoints/roberta.base/model DATA=wndef1000 ARCH=roberta_base MAX_EPOCH=20 MAX_SENTENCES=16 UPDATE_FREQ=64 TOKENS_PER_SAMPLE=128 MAX_POSITIONS=128 source run-sempre_new-roberta.sh

# large-4gpu
# GPU=0,1,2,3 NUM_GPU=4 PORT=67676 MODEL=checkpoints/roberta.large/model DATA=wndef1000 ARCH=roberta_large MAX_EPOCH=20 MAX_SENTENCES=4 UPDATE_FREQ=128 TOKENS_PER_SAMPLE=256 MAX_POSITIONS=256 source run-sempre_new-roberta.sh

# required
MODEL=${MODEL:-"model"}
DATA=${DATA:-""}
ARCH=${ARCH:-"roberta_base"}

# hyperparameters

# each epoch for wn contains 452 batches of 2048 samples
# total updates are for 10 epochs
# warmup updates are 10%
BASE_UPDATES=${BASE_UPDATES:-452}
MAX_EPOCH=${MAX_EPOCH:-10}

LR=${LR:-0.00002}                  # Peak LR for polynomial LR scheduler.
TOKENS_PER_SAMPLE=${TOKENS_PER_SAMPLE:-256}
MAX_POSITIONS=${MAX_POSITIONS:-256}
MAX_SENTENCES=${MAX_SENTENCES:-32} # Batch size.
#MAX_TOKENS=${MAX_TOKENS:-2000}
UPDATE_FREQ=${UPDATE_FREQ:-16}

# default
DATA_ROOT=${DATA_ROOT:-"data-raw"}
if [[ ! -v GPU ]]; then
    GPU=${GPU:-0}
fi
NUM_GPU=${NUM_GPU:-1}
PORT=${PORT:-29500}

# optional
USER_DIR=${USER_DIR:-"../sempre"}
ARGS=${ARGS:-""}
SEED=${SEED:-1234}


# inferred
ID="batchsize$((MAX_SENTENCES*UPDATE_FREQ*NUM_GPU))-seed${SEED}"

if [[ $MAX_EPOCH -ne 10 ]]; then
    ID="${ID}-me${MAX_EPOCH}"
fi

TOTAL_TOTAL_UPDATES=$((BASE_UPDATES*MAX_EPOCH))
TOTAL_NUM_UPDATES=$((BASE_UPDATES*MAX_EPOCH))

if [[ ! -v WARMUP_UPDATES ]]; then
    WARMUP_UPDATES=$((TOTAL_TOTAL_UPDATES/10))
fi

CHECKPOINT_ROOT="${MODEL}-new/$ID"

if [[ ! -z "$USER_DIR" ]]; then
    ARGS="${ARGS} --user-dir ${USER_DIR} "
fi

if [[ -z "$GPU" ]]; then
    ARGS="${ARGS} --cpu "
    CMD="fairseq-train"
    MAX_SENTENCES=$((MAX_SENTENCES*UPDATE_FREQ))
    UPDATE_FREQ=1
else
    if [[ $NUM_GPU -gt 1 ]]; then
        CMD="python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT -m fairseq_cli.train "
        ARGS="${ARGS} --ddp-backend=no_c10d --distributed-no-spawn"
    else
        CMD="fairseq-train"
    fi
fi

echo "running command: $CMD"
echo "running model ID: $ID"
echo "running extra args: $ARGS"


if [[ -f "$CHECKPOINT_ROOT/checkpoint_last.pt" ]]; then
    # continue training
    CUDA_VISIBLE_DEVICES=$GPU $CMD "$DATA_ROOT/$DATA" \
        $ARGS \
        --required-batch-size-multiple 2 \
        --task masked_lm_prediction --criterion masked_lm_prediction \
        --separator-token "</s>" --init-token "<s>" --mask-token "<mask>" --def-sep-token "<:>" --surround-token "<sur>"
        --num-workers 8 --tokens-per-sample $TOKENS_PER_SAMPLE --max-positions $MAX_POSITIONS \
        --mask-prob 0.15 --leave-unmasked-prob 0.1 --random-token-prob 0.1 \
        --bpe gpt2 --arch $ARCH \
        --activation-fn gelu --pooler-activation-fn tanh \
        --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 --pooler-dropout 0.1 \
        --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 --weight-decay 0.01 \
        --lr-scheduler polynomial_decay --lr $LR \
        --max-epoch $MAX_EPOCH  --max-update $TOTAL_NUM_UPDATES \
        --total-num-update $TOTAL_TOTAL_UPDATES --warmup-updates $WARMUP_UPDATES \
        --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
        --skip-invalid-size-inputs-valid-test \
        --save-dir "$CHECKPOINT_ROOT" --seed $SEED \
        --save-interval 1 --keep-last-epochs 10 \
        --log-format simple --log-interval 10 |
        tee -a "$CHECKPOINT_ROOT/${ID}-train.txt"
else
    # not started, run training
    mkdir -p "$CHECKPOINT_ROOT"

    CUDA_VISIBLE_DEVICES=$GPU $CMD "$DATA_ROOT/$DATA" \
        $ARGS \
        --restore-file "${MODEL}.pt" \
        --reset-optimizer --reset-dataloader --reset-meters \
        --required-batch-size-multiple 2 \
        --task masked_lm_prediction --criterion masked_lm_prediction \
        --separator-token "</s>" --init-token "<s>" --mask-token "<mask>" --def-sep-token "<:>" --surround-token "<sur>"
        --num-workers 8 --tokens-per-sample $TOKENS_PER_SAMPLE --max-positions $MAX_POSITIONS \
        --mask-prob 0.15 --leave-unmasked-prob 0.1 --random-token-prob 0.1 \
        --bpe gpt2 --arch $ARCH \
        --activation-fn gelu --pooler-activation-fn tanh \
        --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 --pooler-dropout 0.1 \
        --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 --weight-decay 0.01 \
        --lr-scheduler polynomial_decay --lr $LR \
        --max-epoch $MAX_EPOCH  --max-update $TOTAL_NUM_UPDATES \
        --total-num-update $TOTAL_TOTAL_UPDATES --warmup-updates $WARMUP_UPDATES \
        --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
        --skip-invalid-size-inputs-valid-test \
        --save-dir "$CHECKPOINT_ROOT" --seed $SEED \
        --save-interval 1 --keep-last-epochs 10 \
        --log-format simple --log-interval 10 |
        tee "$CHECKPOINT_ROOT/${ID}-train.txt"
fi