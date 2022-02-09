#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

PROJ_ROOT="/data1/junbao/RDA-git"
ALGORITHM="RDA"
PROJ_NAME="A2D"
SOURCE="webcam"
TARGET="amazon"
NOISY_TYPE="feature_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DEL_RATE="0"
DATASET="Office-31"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

#echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/DANN_train.py \
    --config ${PROJ_ROOT}/config/tcl.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
#    >> ${LOG_FILE}  2>&1
