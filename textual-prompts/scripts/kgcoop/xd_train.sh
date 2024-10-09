#!/bin/bash

# cd ../..

# custom config
DATA=/DATA
TRAINER=KgCoOp
WEIGHT=8.0
CFG=vit_b16_ep10_bt4_cross_dataset
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots 
CSC=False  # class-specific context (False or True)

for SHOTS in 16
do
for DATASET in imagenet
do
for SEED in 1 2 3
do
    DIR=output/imagenet/${TRAINER}/${CFG}_16shots/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        CUDA_VISIBLE_DEVICES=1 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W ${WEIGHT} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done
done
done
