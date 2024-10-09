#!/bin/bash

#cd ../..

# custom config
DATA="your_data_path"
TRAINER=ProDA

# DATASET=$1
# SEED=$2
# DEVICE=$3
# BEST=$4

CFG=vit_b16_ep50_c4_BZ4_ProDA
SHOTS=16
LOADEP=50
# SUB=$2
DEVICE=$1

for DATASET in caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets stanford_cars sun397 ucf101
do
    for SUB in new1 new2 new3 new4 new5 new_ratio1 new_ratio2 new_ratio3 new_ratio4 new_ratio5
    do
        for SEED in 1 2 3
        do
            COMMON_DIR=${DATASET}/${CFG}/seed${SEED}
            MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
            DIR=output/base2new/test_${SUB}/${COMMON_DIR}
            if [ -d "$DIR" ]; then
                echo "Evaluating model"
                echo "Results are available in ${DIR}. Resuming..."

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
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES ${SUB}

            else
                echo "Evaluating model"
                echo "Runing the first phase job and save the output to ${DIR}"

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
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES ${SUB}
            fi
        done
    done
done