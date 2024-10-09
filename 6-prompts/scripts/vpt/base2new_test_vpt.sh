#!/bin/bash

#cd ../..

# custom config
DATA="your_data_path"
TRAINER=VPT

# DATASET=$1
# SEED=$2
DEVICE=$1
CFG=vit_b16_c4_ep50_batch4_4
SHOTS=16
LOADEP=50
# SUB=new

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
                echo "Evaluating model"
                echo "Results are available in ${DIR}. Resuming..."

                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --model-dir ${MODEL_DIR} \
                --load-epoch ${LOADEP} \
                --eval-only \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES ${SUB}

            else
                echo "Evaluating model"
                echo "Runing the first phase job and save the output to ${DIR}"

                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --model-dir ${MODEL_DIR} \
                --load-epoch ${LOADEP} \
                --eval-only \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES ${SUB}
            fi
        done
    done
done