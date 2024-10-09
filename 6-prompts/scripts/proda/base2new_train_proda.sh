#!/bin/bash

#cd ../..

# custom config
DATA="your_data_path"
TRAINER=ProDA

# DATASET=$1
# SEED=$3
# DEVICE=$2
CFG=vit_b16_ep50_c4_BZ4_ProDA #vit_b16_c2_e100_ctx16_warmup5_BZ64_ProDA
SHOTS=16
DEVICE=$1
for DATASET in caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets stanford_cars sun397 ucf101
do
    for SEED in 1 2 3
    do
        DIR=output/base2new/train_base/${DATASET}/${CFG}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Resuming..."
            CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base
        else
            echo "Run this job and save the output to ${DIR}"
            CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base
        fi
    done
done
