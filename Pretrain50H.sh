#!/bin/bash

echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes: $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

archs=("resnet50")
datasets=("cifar10" "cifar100" "stl10")
epochs=(200 400)
batch_sizes=(256 1024)

echo "Pretrian task begin"

for arch in "${archs[@]}"; do
    for dataset in "${datasets[@]}"; do
        for epoch in "${epochs[@]}"; do
            for batch_size in "${batch_sizes[@]}"; do
                python pretrain.py -data ./datasets -dataset_name ${dataset} --arch ${arch} --epochs ${epoch} --batch_size ${batch_size} --seed 2024
            done
        done
    done
done

echo "Pretrian task end"