#!/bin/bash

#cd ../..

# custom config
DATA="your_data_path"
TRAINER=RPO

# DATASET=$1
# SEED=$2
GPU=$1

CFG=main_K24_ep50_batch4
SHOTS=16
for DATASET in ucf101 sun397 stanford_cars oxford_pets oxford_flowers
do
    for SEED in 1 2 3
    do
        DIR=output/rpo/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        if [ -d "$DIR" ]; then
           echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            CUDA_VISIBLE_DEVICES=${GPU} python train.py \
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