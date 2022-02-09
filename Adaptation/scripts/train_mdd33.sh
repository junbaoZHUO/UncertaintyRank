#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

PROJ_ROOT="/data1/junbao/RDA"
ALGORITHM="MDD"
PROJ_NAME="A2W"
SOURCE="webcam"
TARGET="dslr"
NOISY_TYPE="feature" #feature_feature_feature_uniform, pair, none
NOISY_RATE="0.4"
#DEL_RATE="0.2"
DATASET="Office-31"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

#echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python -W ignore trainer/RDA_train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    # --src_address ../code2/office_list/amazon_list_corrupted_0.4_relabel.txt \
    # --src_address ../code2/office_list/webcam_list_noisy_0.4_relabel_0.7.txt \
    # --tgt_address ../code2/office_list/dslr_list.txt \
 #   >> ${LOG_FILE}  2>&1
