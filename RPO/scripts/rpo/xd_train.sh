#!/bin/bash

# cd ..

# custom config
DATA="your_data_path"
TRAINER=RPO

DATASET=imagenet
SEED=$1
DEVICE=$2

CFG=main_K24_ep10_batch4_cross_dataset
SHOTS=16  # number of shots (1, 2, 4, 8, 16)


DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} 
fi