# required
MODEL=${MODEL:-"model"}
if [[ -f "${MODEL}.pt" ]]; then
    RESTORE="--restore-file ${MODEL}.pt"
else
    echo "model not found or not supported"
    # handle runing with source
    return 1 2>/dev/null || exit 1
fi


TASK="PIQA"
TASK_CMD='piqa'
BPE="gpt2"
INIT_TOKEN="<s>"
SEP_TOKEN="</s>"
VAL_SUBSET="valid"
TEST_SUBSET="tests"
CRITERION='sentence_ranking'
DATA_ROOT="data-raw/PIQA/"


# hyperparameters
# PIQA: 1 epoch consists of 504 batches of 32 examples
ARCH=${ARCH:-"roberta_large"}
LR=${LR:-1e-5}
MAX_EPOCH=${MAX_EPOCH:-50}
TOTAL_NUM_UPDATES=${TOTAL_NUM_UPDATES:-25200}
WARMUP_UPDATES=${WARMUP_UPDATES:-2520}
MAX_SENTENCES=${MAX_SENTENCES:-4}
UPDATE_FREQ=${UPDATE_FREQ:-8}
SEED=${SEED:-9100}
ACT=${ACT:-"tanh"}
PATIENCE=${PATIENCE:-10}
 MAX_POSITIONS=${MAX_POSITIONS:-128}

# default
if [[ ! -v GPU ]]; then
    GPU=${GPU:-0}
fi
USER_DIR=${USER_DIR:-"../sempre"}
ARGS=${ARGS:-""}

# inferred
BATCH_SIZE=$((MAX_SENTENCES * UPDATE_FREQ))
ID="${TASK}-batchsize${BATCH_SIZE}-lr${LR}-seed${SEED}"
if [[ $MAX_EPOCH -ne 10 ]]; then
    ID="${ID}-me${MAX_EPOCH}"
fi
if [[ "$ACT" != "tanh" ]]; then
    ID="${ID}-${ACT}"
fi

if [[ ! -z "$USER_DIR" ]]; then
    ARGS="${ARGS} --user-dir ${USER_DIR} "
fi

if [[ -z "$GPU" ]]; then
    ARGS="${ARGS} --cpu "
    MAX_SENTENCES=$BATCH_SIZE
    UPDATE_FREQ=1
fi

CHECKPOINT_ROOT="$MODEL/$ID"

BEST_ARGS="--best-checkpoint-metric accuracy --maximize-best-checkpoint-metric "


echo "running model ID: $ID"
echo "running extra args: $ARGS"

# activate conda if available
if [[ -v CONDA_SHLVL ]]; then
    if [[ $CONDA_SHLVL -eq 0 ]]; then
        eval "$(conda shell.bash hook)"
        conda activate
        if [[ $? -eq 0 ]]; then
            echo "activated conda"
        else
            echo 'conda activation error'
        fi
    else
        echo 'conda already activated'
    fi
fi

# check if is trained
grep "done training" "$CHECKPOINT_ROOT/${ID}-train.txt"
if [[ $? -ne 0 ]]; then
    # not trained
    rm -r "$CHECKPOINT_ROOT"

    # run training
    mkdir -p "$CHECKPOINT_ROOT"
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        "$DATA_ROOT" $ARGS $RESTORE \
        --max-positions $MAX_POSITIONS \
        --max-sentences $MAX_SENTENCES \
        --update-freq $UPDATE_FREQ \
        --max-tokens 4400 \
        --task $TASK_CMD \
        --reset-optimizer --reset-dataloader --reset-meters \
        --required-batch-size-multiple 4 \
        --bpe "$BPE" --init-token "$INIT_TOKEN" --separator-token "$SEP_TOKEN" \
        --arch $ARCH \
        --criterion $CRITERION \
        --dropout 0.1 --attention-dropout 0.1 \
        --pooler-activation-fn $ACT \
        --weight-decay 0.1 --optimizer "adam" --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
        --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
        --max-epoch $MAX_EPOCH --skip-invalid-size-inputs-valid-test --patience $PATIENCE \
        --find-unused-parameters --seed $SEED \
        --train-subset train --valid-subset ${VAL_SUBSET} \
        $BEST_ARGS --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
        --save-dir "$CHECKPOINT_ROOT" | tee "$CHECKPOINT_ROOT/${ID}-train.txt"
fi

do_validate() {
    local SPLIT=$1
    local VALID_ARGS=""
    local SUBSET=""

    if [[ ! -f ${DATA_ROOT}${SPLIT}.jsonl ]]; then
        echo "validate: ${SPLIT} split not found"
    else
        SUBSET=$SPLIT
    fi

    # if [[ $SPLIT == 'val' ]]; then
    #     SUBSET=$VAL_SUBSET
    # elif [[ $SPLIT == 'test' ]]; then
    #     SUBSET=$TEST_SUBSET
    # else
    #     echo "validate: ${SPLIT} split not found"
    #     return 1
    # fi

    grep "done" "$CHECKPOINT_ROOT/${ID}-$SPLIT.txt"
    if [[ $? -ne 0 ]]; then
        # not evaluated
        if [[ ! -z "$USER_DIR" ]]; then
            VALID_ARGS="$VALID_ARGS --user-dir $USER_DIR"
        fi

        if [[ -z "$GPU" ]]; then
            VALID_ARGS="$VALID_ARGS --cpu"
        fi

        CUDA_VISIBLE_DEVICES=$GPU python inference.py \
            "$DATA_ROOT" \
            --checkpoint-dir "$CHECKPOINT_ROOT" \
            --checkpoint-name "checkpoint_best.pt" \
            --mode "validate" \
            --valid-subset $SUBSET \
            --task $TASK \
            $VALID_ARGS |
            tee "$CHECKPOINT_ROOT/${ID}-${SPLIT}.txt"

    fi
    return 0
}

do_test() {
    local SUBSET=$1
    local VALID_ARGS=""
    local MODE=""
    local OUT=""

    if [[ -f ${DATA_ROOT}/${SUBSET}.jsonl ]]; then
        MODE=${DATA_ROOT}/${SUBSET}.jsonl
        OUT=${CHECKPOINT_ROOT}/${TASK}.jsonl
    else
        echo "test: split with subset ${SUBSET} not found"
        return 1
    fi

    if [[ ! -f "${CHECKPOINT_ROOT}/${TASK}.jsonl" ]] || [[ $(stat -c%s "${CHECKPOINT_ROOT}/${TASK}.jsonl") -eq 0 ]]; then
        # not tested
        if [[ ! -z "$USER_DIR" ]]; then
            VALID_ARGS="$VALID_ARGS --user-dir $USER_DIR"
        fi

        if [[ -z "$GPU" ]]; then
            VALID_ARGS="$VALID_ARGS --cpu"
        fi

        CUDA_VISIBLE_DEVICES=$GPU python inference.py \
            --checkpoint-dir "$CHECKPOINT_ROOT" \
            --checkpoint-name "checkpoint_best.pt" \
            --mode $MODE \
            --task $TASK \
            $VALID_ARGS \
            --save-path $OUT \
            "${DATA_ROOT}"
    fi
    return 0
}

if [[ -f "${CHECKPOINT_ROOT}/checkpoint_best.pt" ]]; then

    do_validate $VAL_SUBSET
    do_test $TEST_SUBSET

    # SHOULD_REMOVE=1
    # # valdiation fail
    # if [[ -f "$CHECKPOINT_ROOT/${ID}-val.txt" ]]; then
    #     grep "done" "$CHECKPOINT_ROOT/${ID}-val.txt"
    #     if [[ $? -ne 0 ]]; then
    #         SHOULD_REMOVE=0
    #     fi
    # else
    #     SHOULD_REMOVE=0
    # fi

    # if [[ -f "${CHECKPOINT_ROOT}/${TASK}.jsonl" ]]; then

    #     SIZE=$(stat -c%s "${CHECKPOINT_ROOT}/${TASK}.jsonl")
    #     if [[ $SIZE -eq 0 ]]; then
    #         SHOULD_REMOVE=0
    #     fi
    # fi

    # if [[ SHOULD_REMOVE -eq 1 ]]; then
    #     rm $CHECKPOINT_ROOT/checkpoint_best.pt
    # fi

fi

# deactivate conda if activated
# if [[ -v CONDA_SHLVL ]]; then
#     if [[ $CONDA_SHLVL -eq 1 ]]; then
#         conda deactivate
#         echo "deactivated conda"
#     fi
# fi
