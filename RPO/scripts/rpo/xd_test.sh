#!/bin/bash

cd ../..

# custom config
DATA="your_data_path"
TRAINER=TCP
# TRAINER=CoOp

DATASET=$1
SEED=$2

CFG=vit_b16_c4_ep10_batch4_cross_dataset
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
WEIGHT=8.0

DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch 10 \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.W ${WEIGHT} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} 
fi