#!/bin/bash

# cd ..

# custom config
DATA="/mnt/hdd/DATA"
TRAINER=ProGrad

# DATASET=$1
CFG=vit_b16_ep50_batch4_c4  # config file
CTP=end  # class token position (end or middle)
NCTX=4 # number of context tokens
SHOTS=16 # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
LAMBDA=0.8
DEVICE=$1
LOADEP=50
# SUB=$2
for DATASET in caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets stanford_cars sun397 ucf101
do
    for SUB in new1 new2 new3 new4 new5 new_ratio1 new_ratio2 new_ratio3 new_ratio4 new_ratio5
    do
        for SEED in 1 2 3
        do
            COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
            MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
            DIR=output/base2new/test_${SUB}/${COMMON_DIR}
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
                --model-dir ${MODEL_DIR} \
                --load-epoch ${LOADEP} \
                --eval-only \
                LOSS.LAMBDA ${LAMBDA} \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.COOP.CSC ${CSC} \
                TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES ${SUB}
            fi
        done
    done
done
