#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

PROJ_ROOT="/data1/junbao/TMM-git/Adaptation"
ALGORITHM="MDD"
PROJ_NAME="A2W"
SOURCE="Art"
TARGET="Product"
NOISY_TYPE="feature_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DATASET="Office-Home"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

#echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python -W ignore trainer/train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --maxIter 40000 \
    # --src_address ../code2/office_list/amazon_list_corrupted_0.4_relabel.txt \
    # --src_address ../code2/office_list/webcam_list_noisy_0.4_relabel_0.7.txt \
    # --tgt_address ../code2/office_list/dslr_list.txt \
 #   >> ${LOG_FILE}  2>&1