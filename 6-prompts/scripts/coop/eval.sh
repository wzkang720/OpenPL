#!/bin/bash

#cd ../..

# custom config
DATA="/DATA"
TRAINER=CoOp
SHOTS=16
NCTX=4
CSC=False
CTP=end

# DATASET=$1
# sub=$2
DEVICE=$1
CFG=vit_b16_ep50_BS4
for DATASET in caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets stanford_cars sun397 ucf101
do
    for sub in new1 new2 new3 new4 new5 new_ratio1 new_ratio2 new_ratio3 new_ratio4 new_ratio5
    do
        for SEED in 1 2 3
        do
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir output/evaluation/${sub}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED} \
            --model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
            --load-epoch 50 \
            --eval-only \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.SUBSAMPLE_CLASSES ${sub}
        done
    done
done
