#!/bin/bash

# cd ..

# custom config
DATA="your_data_path"
TRAINER=ProGrad

# DATASET=$1
CFG=vit_b16_ep50_batch4_c4_cross_dataset  # config file
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
LAMBDA=0.8
DEVICE=$1
for DATASET in imagenet
do
    for SEED in 1 2 3
    do
        DIR=output/imagenet/${TRAINER}/${CFG}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Skip this job"
        else
            echo "Run this job and save the output to ${DIR}"
            CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            LOSS.LAMBDA ${LAMBDA} \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS} 
        fi
    done
done
