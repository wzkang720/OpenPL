GPU=$1
SHOT=16

for dataset in eurosat dtd fgvc_aircraft oxford_flowers stanford_cars oxford_pets food101 sun397 ucf101 caltech101 imagenet
do
    for seed in 1 2 3
    do
        # training
        sh scripts/cocoop/base2new_train.sh ${dataset} ${seed} ${SHOT} ${GPU}
        # evaluation
        sh scripts/cocoop/base2new_test.sh ${dataset} ${seed} ${SHOT} base ${GPU}
        sh scripts/cocoop/base2new_test.sh ${dataset} ${seed} ${SHOT} new ${GPU} 
    done
done