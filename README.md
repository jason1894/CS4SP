# Deep Contrastive Learning for Supervised  Learning

This project contains the experiments for CL4SP, which has two stages: pretraining and downstream tasks. The first pretraining stage is to learn a good representation of the data. Secondly, the learned representations are used for supervised tasks like regression and classification.


## Preparations
```
# install dependent packages
pip install -r requiremets.txt
```

## Pretraining
To save time and facilitate result reproduction, we provide the pretrained model checkpoints in [checkpoints](https://drive.google.com/drive/folders/1jlc0MblXDMosu9r8RGf-fqOvGvUeHDBe?usp=sharing)

If you want to train the representation models, you can run
```
 python ./pretrain.py -data ./datasets -dataset_name "cifar10" --arch "resnet18" --epochs 200 --batch_size 256 --seed 2024
```
Which will get the related  pretrained model.
However, we provide bash scripts for running tasks, such as Pretrain18H.sh, which will produce all pretrained models based on the resnet18 network,
```
./Pretrain18H.sh
```
All pretrained models are saved in folder "PretrainMole". You can check the training log files in there.

## Downstream tasks

With pretrained model in hand, you can go to downstream tasks, like
```
 python eval.py -data ./datasets -dataset_name "cifar10" -s 0.1 --arch "resnet18" --epochs 100 --batch_size 16 --mepochs 200 --mbatch_size 256 --nclass 10 --nloop 1 --seed 2024
```

There are bash scripts for bath processing tasks
```
./CifarEvalH.sh
```

We also added imbalanced dataset experiments for downstream tasks to evaluate the robustness of the model. Run the program the same way as before.

## Results
- Feature visuliazation 

<img src="https://github.com/jason1894/CS4SP/blob/master/plots/features.png" alt="Feature Visualization" title="CIFAR-10">

- Top-1 Accuracy

<img src="https://github.com/jason1894/CS4SP/blob/master/plots/results.png" alt="Top-1 accuracy" title="Results">



