#!/bin/bash

#cd ../..

# custom config
DATA="your_data_path"
TRAINER=ZeroshotCLIP
# DATASET=$1
CFG=vit_b16_ep50_bs4  # rn50, rn101, vit_b32 or vit_b16
# sub=$2
# ImageNetR ImageNetA ImageNetV2 ImageNetSketch
# Caltech101 DescribableTextures EuroSAT FGVCAircraft Food101 OxfordFlowers OxfordPets StanfordCars SUN397 UCF101
DEVICE=$1
for DATASET in Caltech101 DescribableTextures EuroSAT FGVCAircraft Food101 OxfordFlowers OxfordPets StanfordCars SUN397 UCF101
do
    for sub in base new_ratio1 new_ratio2 new_ratio3 new_ratio4 new_ratio5
    do
        for SEED in 1 2 3
        do
            CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
            --root ${DATA} \
            --trainer ${TRAINER} \
            --seed ${SEED} \
            --dataset-config-file configs/datasets/imagenet.yaml \
            --config-file configs/trainers/CoOp/${CFG}.yaml \
            --output-dir output/evaluation/${TRAINER}/${CFG}/${DATASET}/${sub}/seed${SEED} \
            --dataset_var ${DATASET}\
            --eval-only\
            DATASET.SUBSAMPLE_CLASSES ${sub}
        done
    done
done