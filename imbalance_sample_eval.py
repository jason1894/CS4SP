"""
This script is for imbalance sample eval.
"""

import logging
import os
import csv
from tqdm import trange
import yaml

import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torch.cuda.amp import GradScaler
from torchmetrics.classification import MulticlassF1Score
from data_aug.cl_dataset import CLDataset, set_seed


models_names = sorted(name for name in models.__dict__
                      if name.islower() and not name.startswith("__")
                      and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Pytorch BniaryCLR')
parser.add_argument('-data', metavar='DOR', default='./datasets', 
                    help='path to dataset')
parser.add_argument('-dataset_name', default='cifar10',
                    help='dataset name',)
parser.add_argument('-s', '--split', default=1.0, type=float,
                    help="the ratio of of the data to use, max = 1.0")
parser.add_argument('-ra', '--randoms', default= True, type=bool, choices=[True, False],
                    help="wheture to random split")
parser.add_argument('--nclass', default=10, type= int, 
                    help="the output class")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',choices=models_names,
                    help='model architecture: ' +
                            ' | '.join(models_names) +
                            ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epochs', default=200, type = int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--mepochs', default=200, type = int, metavar='N',
                    help='number of total epochs of trained modal')
parser.add_argument('-mb', '--mbatch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size trainded modal')

parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float, metavar='LR',
                    help='initial learning rate',dest='lr')
parser.add_argument('--wd', '--weight-decay', default=8e-4, type=float,
                    metavar='W', help='weight decay (default: 8e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--device',default= torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--nloop', default=1, type=int, metavar='N',
                    help='Number of train loops contrastive learning training.')

def imbalance_set(dataset, class_unbalance_ratio, splits):

    if  hasattr(dataset, 'targets'):
        targets_daset = dataset.targets
    elif hasattr(dataset, 'labels'):
        targets_daset = dataset.labels
    elif hasattr(dataset, '_labels'): 
        targets_daset = dataset._labels  
    else:
        raise ValueError("please check the data targets/labels attribute")
    
    class_num = len(set(targets_daset))    
    class_indices = {}
    for i in range(class_num):
        class_indices[i] = [j for j, label in enumerate(targets_daset) if label == i]

    new_train_indices = []
    for i in range(class_num):
        num_samples = int(class_unbalance_ratio[i] * len(class_indices[i]))
        selected_indices = torch.randint(0, len(class_indices[i]), (num_samples,))
        new_train_indices.extend([class_indices[i][j] for j in selected_indices])  

    unbalance_subset = Subset(dataset, new_train_indices)
    unbalance_trainsubset, unbalance_validsubset  =random_split(unbalance_subset, [splits, 1-splits])
    return unbalance_trainsubset, unbalance_validsubset    

def F1score(logits, y_true, device):  
    nclass = logits.shape[1]
    metrix = MulticlassF1Score(nclass, average='macro').to(device)
    f1 = metrix(logits, y_true)
    return f1       



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(model, data_train_loader, data_test_loader, args, name):
    scaler = GradScaler(enabled=args.fp16_precision)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    model = model.to(args.device)
    result = []
    
    filearg = "{}_ep{:04}bt{}sp{}".format(args.arch, args.mepochs, args.mbatch_size, args.split) #may change by your need
    log_dir = os.path.join("results_imbana", args.dataset_name, "multiclass", str(args.nloop), filearg)
    writer = SummaryWriter(log_dir=log_dir)
    # Todo save the config file
    n_iter = 0

    logging.info(f"star the downstream task for {args.epochs} epochs.")
    logging.info(f"Training with not gpu: {args.disable_cuda}.")

    for epoch in trange(args.epochs):
        top_train_accuracy = 0
        counter_train =0
        loss_epoch = 0
        model.train()

        for counter_train, (x, y) in enumerate(data_train_loader):
            # x_batch = x[0].to(args.device)
            # 多分类
            x_batch = x.to(args.device)
            y_batch = y.to(args.device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch = loss.item()
        writer.add_scalar("{}_trian loss".format(name), loss_epoch, epoch)    
            
        top_train_accuracy /= (counter_train +1)

        model.eval()
        top_test_accuracy = 0 
        f1_score_test= 0 
        counter_test= 0                     
        for counter_test, (x, y) in enumerate(data_test_loader):
            # x_batch = x[0].to(args.device)
            # 多分类
            x_batch = x.to(args.device)
            y_batch = y.to(args.device)

            logits = model(x_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            f1_score = F1score(logits, y_batch, args.device)
            top_test_accuracy += top1[0]
            f1_score_test +=  f1_score


        top_test_accuracy/=(counter_test + 1) 
        f1_score_test   /= (counter_test + 1) 

        result.append([epoch, top_train_accuracy.item(), top_test_accuracy.item(), f1_score_test.item()])
        writer.add_scalar("{}_top1 train acc".format(name), top_train_accuracy.item(), epoch)
        writer.add_scalar("{}_top1 test acc".format(name), top_test_accuracy.item(), epoch)
        writer.add_scalar("{}_f1 score".format(name), f1_score_test.item(), epoch)
    
    # save the evalue data to csv for futher data analyse.
    filename = os.path.join(writer.log_dir, "{}{}.csv".format(name, filearg))
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Epoch', 'Top1 train accuracy', 'Top1 test accuracy', "F1 score"])
        csvwriter.writerows(result)


def main():
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.decive = torch.device('cpu')
        args.gpu_index = -1    

    """ reload pretrain model and supervised moadel """
    if args.arch == 'resnet18':
        clmodel = models.resnet18(weights= None, num_classes=args.nclass)

        supmodel = models.resnet18(weights= None, num_classes=args.nclass)

    elif args.arch == 'resnet50':   
        clmodel = models.resnet50(weights= None, num_classes=args.nclass)

        supmodel = models.resnet50(weights= None, num_classes=args.nclass)
    else:
        raise ValueError("the model must be resnet18 or resnet50 ")

    # reload the checkpoint
    path1 = "{}_ep{:04}bt{}".format(args.arch, args.mepochs, args.mbatch_size)
    filename = "CP_{}ep{:04}bt{}.pth.tar".format(args.arch, args.mepochs, args.mbatch_size)
    checkpointpath = os.path.join("PretrainModel", args.dataset_name, path1, filename)

    checkpoint = torch.load(checkpointpath, map_location= args.device)  
      
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]

    log = clmodel.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']

    # freeze all layers but the last fc
    for name, param in clmodel.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    parameters = list(filter(lambda p: p.requires_grad, clmodel.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias


    """dataset reload"""
    dataset = CLDataset(args.data, False)
    TrmTrainset, TrmTestset = dataset.get_dataset(args.dataset_name)
    
    # 多分类 unbalance subset
    # trian data 
    with open('imbalance_config.yaml', 'r') as file:
        yaml_config = yaml.safe_load(file)   
    setattr(args, "unbalance_ratio", yaml_config.get('config3'))

    Binary_train, _ = imbalance_set(TrmTrainset, args.unbalance_ratio, args.split)
    Binary_test = TrmTestset

    print("len of train:", Binary_train.__len__())
    print("len of test:", Binary_test.__len__())
    Binary_train_Loader = DataLoader(Binary_train, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers,
                                     pin_memory=True, drop_last= False)
    
    Binary_test_Loader = DataLoader(Binary_test, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers,
                                     pin_memory=True, drop_last= False)   

    """ downstream train and test"""
    # CL
    print("CL task")
    train(clmodel, Binary_train_Loader, Binary_test_Loader, args, name = "CL")

    # SUP
    print("super task")
    train(supmodel, Binary_train_Loader, Binary_test_Loader, args, name = "SP")


if __name__ == "__main__":
    main()
