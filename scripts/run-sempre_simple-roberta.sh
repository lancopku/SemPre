# script for runnig unpaired word-definitions and in terms of the pos tag of the word

# base-1gpu
# GPU=0 NUM_GPU=1 MODEL=checkpoints/base POS=x ARCH=roberta_base MAX_EPOCH=10 MAX_SENTENCES=32 UPDATE_FREQ=64 TOKENS_PER_SAMPLE=128 MAX_POSITIONS=128 source run-sempre_simple-roberta.sh

# large-2gpu
# GPU=0,1 NUM_GPU=2 PORT=67676 MODEL=checkpoints/base POS=x ARCH=roberta_large MAX_EPOCH=10 MAX_SENTENCES=8 UPDATE_FREQ=128 TOKENS_PER_SAMPLE=256 MAX_POSITIONS=256 source run-sempre_simple-roberta.sh

# required
MODEL=${MODEL:-"model"}
DATA=${DATA:-""}
ARCH=${ARCH:-"roberta_base"}
POS=${POS:-"x"}

# hyperparameters

# 206878 lemmas
#  146347 are nouns
#  25047 are verbs
#  35584 are adjectives or adverbs
# each epoch for wn contains 452 batches of 2048 samples
# total updates are for 10 epochs
# warmup updates are 10%
if [[ "$POS" == "x" ]]; then
    BASE_UPDATES=102
elif [[ "$POS" == 'n' ]]; then
    BASE_UPDATES=72
elif [[ "$POS" == 'v' ]]; then
    BASE_UPDATES=13
elif [[ "$POS" == 'a' ]]; then
    BASE_UPDATES=18
else
    BASE_UPDATES=452
fi

#BASE_UPDATES=${BASE_UPDATES:-452}
MAX_EPOCH=${MAX_EPOCH:-10}

LR=${LR:-0.00002}                  # Peak LR for polynomial LR scheduler.
TOKENS_PER_SAMPLE=${TOKENS_PER_SAMPLE:-256}
MAX_POSITIONS=${MAX_POSITIONS:-256}
MAX_SENTENCES=${MAX_SENTENCES:-32} # Batch size.
MAX_TOKENS=${MAX_TOKENS:-4400}
UPDATE_FREQ=${UPDATE_FREQ:-16}
CUSTOM_ID=${CUSTOM_ID:-""}
# default
DATA=${DATA:-"data-bin/bookcorpus_all"}
#DATA_ROOT=${DATA_ROOT:-"data-raw"}
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
if [[ ! -z "$CUSTOM_ID" ]]; then
    ID="${CUSTOM_ID}-batchsize$((MAX_SENTENCES*UPDATE_FREQ*NUM_GPU))-seed${SEED}"
else
ID="batchsize$((MAX_SENTENCES*UPDATE_FREQ*NUM_GPU))-seed${SEED}"
fi

if [[ "$POS" != "x" ]]; then
    ID="${ID}-p${POS}"
fi


if [[ $MAX_EPOCH -ne 10 ]]; then
    ID="${ID}-me${MAX_EPOCH}"
fi


TOTAL_TOTAL_UPDATES=$((BASE_UPDATES*MAX_EPOCH))
TOTAL_NUM_UPDATES=$((BASE_UPDATES*MAX_EPOCH))
if [[ ! -v WARMUP_UPDATES ]]; then
    WARMUP_UPDATES=$((TOTAL_TOTAL_UPDATES/10))
fi

CHECKPOINT_ROOT="$MODEL-simple/$ID"

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
    CUDA_VISIBLE_DEVICES=$GPU $CMD "$DATA" \
        $ARGS \
        --required-batch-size-multiple 2 \
        --task masked_simple_def_lm --criterion masked_lm \
        --separator-token "</s>" --init-token "<s>" --mask-token "<mask>" --def-sep-token "<:>" --surround-token "<sur>" \
        --pos ${POS} \
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
        --save-interval 1 --keep-last-epochs 10 --disable-validation \
        --log-format simple --log-interval 10 |
        tee -a "$CHECKPOINT_ROOT/${ID}-train.txt"
else
    # not started, run training
    mkdir -p "$CHECKPOINT_ROOT"

    CUDA_VISIBLE_DEVICES=$GPU $CMD "$DATA" \
        $ARGS \
        --restore-file "${MODEL}.pt" \
        --reset-optimizer --reset-dataloader --reset-meters \
        --required-batch-size-multiple 2 \
        --task masked_simple_def_lm --criterion masked_lm \
        --separator-token "</s>" --init-token "<s>" --mask-token "<mask>" --def-sep-token "<:>" --surround-token "<sur>" \
        --pos ${POS} \
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
        --save-interval 1 --keep-last-epochs 10 --disable-validation \
        --log-format simple --log-interval 10 |
        tee "$CHECKPOINT_ROOT/${ID}-train.txt"
fi