import os
import shutil

# Ensure that the required environment variables are set
DATA="your_data_path"
TRAINER='ProDA'
SHOTS=16
NCTX=4
CSC=False
CTP='end'

# MaPLe
# CFG = 'vit_b16_c4_ep50_batch4'
# ZeroshotCLIP
# CFG='vit_b16_ep50_bs4'
# ProDA
CFG='vit_b16_ep50_c4_BZ4_ProDA'

DATASETS = ['caltech101','dtd','eurosat','fgvc_aircraft','food101','imagenet','oxford_flowers','oxford_pets','stanford_cars','sun397','ucf101']
SUBS = ['new1','new2','new3','new4','new5','new_ratio1','new_ratio2','new_ratio3','new_ratio4','new_ratio5']

for DATASET in DATASETS:
    for SEED in [1,2,3]:
        # MaPLe
        # base_path = f'output/base2new/train_base/{DATASET}/shots_{SHOTS}/{TRAINER}/{CFG}/seed{SEED}'
        # ZeroshotCLIP
        # base_path = f'output/base/{TRAINER}/{CFG}/{DATASET}/{SEED}'
        # ProDA
        base_path = f'output/base2new/train_base/{DATASET}/{CFG}/seed{SEED}'

        new_path = f"res/{DATASET}/{TRAINER}/base/seed{SEED}"
        os.makedirs(new_path, exist_ok=True)
        shutil.copyfile(base_path+'/log.txt',new_path+'/log.txt')
        for SUB in SUBS:
            # MaPLe
            # origin_path = f'output/base2new/test_{SUB}/{DATASET}/shots_{SHOTS}/{TRAINER}/{CFG}/seed{SEED}'
            # ZeroshotCLIP
            # origin_path = f'output/{SUB}/{TRAINER}/{CFG}/{DATASET}/{SEED}'
            # ProDA
            origin_path = f'output/base2new/test_{SUB}/{DATASET}/{CFG}/seed{SEED}'

            new_path = f"res/{DATASET}/{TRAINER}/{SUB}/seed{SEED}"
            # Create the target directory if it does not exist
            os.makedirs(new_path, exist_ok=True)
            # Copy files from origin_path to new_path
            shutil.copytree(origin_path, new_path, dirs_exist_ok=True)