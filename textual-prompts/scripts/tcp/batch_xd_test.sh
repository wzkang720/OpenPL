#!/bin/bash


# custom config
DATA="your_data_path"
TRAINER=TCP
WEIGHT=8.0
CFG=vit_b16_c4_ep10_batch4_cross_dataset
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots 
CSC=False  # class-specific context (False or True)
DEVICE=$1
# SUB=$2

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

                CUDA_xVISIBLE_DEVICES=${DEVICE} python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/imagenet.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --model-dir ${MODEL_DIR} \
                --load-epoch 10 \
                --dataset_var ${DATASET}\
                --eval-only \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.COOP.CSC ${CSC} \
                TRAINER.COOP.W ${WEIGHT} \
                TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS}\
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
                --load-epoch 10 \
                --dataset_var ${DATASET}\
                --eval-only \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.COOP.CSC ${CSC} \
                TRAINER.COOP.W ${WEIGHT} \
                TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS}\
                DATASET.SUBSAMPLE_CLASSES ${SUB} 
            fi
        done
    done
done