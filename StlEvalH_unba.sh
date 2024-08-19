#!/bin/bash

echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes: $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

archs=("resnet18" "resnet50")
datasets=("stl10")
mepochs=(200)
mbatch_sizes=(256)
splits=(0.1 0.5 1.0)

echo "stl10 evalue task begin"

for arch in "${archs[@]}"; do
    for mepoch in "${mepochs[@]}"; do
        for mbatch_size in "${mbatch_sizes[@]}"; do
            for split in "${splits[@]}"; do
            python imbalance_sample_eval.py -data ./datasets -dataset_name "stl10" -s ${split} --arch ${arch} --epochs 100 --batch_size 16 --mepochs ${mepoch} --mbatch_size ${mbatch_size} --nclass 10 --nloop 1 --seed 2024
            done
        done
    done
done

echo "stl10 evalue task end"