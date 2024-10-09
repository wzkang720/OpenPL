#!/bin/bash


# custom config
DATA="your_data_path"
TRAINER=MaPLe

DEVICE=$1

CFG=vit_b16_c2_ep5_batch4_2ctx_cross_datasets
SHOTS=16
LOADEP=10
# SUB=$2
# ImageNetR ImageNetA ImageNetV2 ImageNetSketch
# Caltech101 DescribableTextures EuroSAT FGVCAircraft Food101 OxfordFlowers OxfordPets StanfordCars SUN397 UCF101
# base new_ratio1 new_ratio2 new_ratio3 new_ratio4 new_ratio5
for DATASET in Caltech101 DescribableTextures EuroSAT FGVCAircraft Food101 OxfordFlowers OxfordPets StanfordCars SUN397 UCF101
do
    for SUB in base new_ratio1 new_ratio2 new_ratio3 new_ratio4 new_ratio5
    do
        for SEED in 1 2 3
        do
        #/prompt-learning-evaluation/6-prompts/output/imagenet/VPT/vit_b16_c4_ep5_batch4_4_cross_dataset_16shots/seed1
            COMMON_DIR=${TRAINER}/${CFG}_16shots/seed${SEED}
            MODEL_DIR=output/imagenet/${TRAINER}/${CFG}_16shots/seed${SEED}
            DIR=output/evaluation/${TRAINER}/${CFG}_16shots/${DATASET}/${SUB}/seed${SEED}
            if [ -d "$DIR" ]; then
                echo "Evaluating model"
                echo "Results are available in ${DIR}. Resuming..."

                CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/imagenet.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --model-dir ${MODEL_DIR} \
                --load-epoch ${LOADEP} \
                --dataset_var ${DATASET}\
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
                --dataset-config-file configs/datasets/imagenet.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --model-dir ${MODEL_DIR} \
                --load-epoch ${LOADEP} \
                --dataset_var ${DATASET}\
                --eval-only \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES ${SUB} 
            fi
        done
    done
done