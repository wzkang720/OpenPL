import os
import shutil

# Ensure that the required environment variables are set
DATA="/mnt/hdd/DATA"
TRAINER='ProGrad'
SHOTS=16
NCTX=4
CSC=False
CTP='end'

CFG='vit_b16_ep50_batch4_c4'
DATASETS = ['caltech101','dtd','eurosat','fgvc_aircraft','food101','imagenet','oxford_flowers','oxford_pets','stanford_cars','sun397','ucf101']
SUBS = ['new1','new2','new3','new4','new5','new_ratio1','new_ratio2','new_ratio3','new_ratio4','new_ratio5']
for DATASET in DATASETS:
    for SEED in [1,2,3]:
        base_path = f'output/base2new/train_base/{DATASET}/shots_{SHOTS}/{TRAINER}/{CFG}/seed{SEED}'
        new_path = f"res/{DATASET}/{TRAINER}/base/seed{SEED}"
        os.makedirs(new_path, exist_ok=True)
        shutil.copyfile(base_path+'/log.txt',new_path+'/log.txt')
        for SUB in SUBS:
            origin_path = f'output/base2new/test_{SUB}/{DATASET}/shots_{SHOTS}/{TRAINER}/{CFG}/seed{SEED}'
            new_path = f"res/{DATASET}/{TRAINER}/{SUB}/seed{SEED}"
            # Create the target directory if it does not exist
            os.makedirs(new_path, exist_ok=True)
            # Copy files from origin_path to new_path
            shutil.copytree(origin_path, new_path, dirs_exist_ok=True)