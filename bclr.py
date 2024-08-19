import logging
import os
from tqdm import trange


import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from utils import save_config_file, accuracy, save_checkpoint


class BCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        termpath = "{}_ep{:04d}bt{}".format(self.args.arch, self.args.epochs,self.args.batch_size)
        log_dir = os.path.join("PretrainModel", self.args.dataset_name, termpath)
        self.writer = SummaryWriter(log_dir=log_dir)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'train.log'), level= logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    
    def nce_bin_loss_MLE_constant(self, features):
        n = features.shape[0]/2
        labels =  torch.cat([torch.arange(n) for i in range(2)] ,dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)
        features = F.normalize(features, dim=1)

        similary = torch.matmul(features, features.T).to(self.args.device)
        mask = torch.eye(labels.shape[0], dtype= torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similary = similary[~mask].view(similary.shape[0], -1)

        postives = similary[labels.bool()].view(labels.shape[0], -1)
        negtives = similary[~labels.bool()].view(similary.shape[0], -1)
        min_neg = torch.min(negtives).item()
        max_neg = torch.max(negtives).item()

        min_pos = torch.min(postives).item()
        max_pos = torch.max(postives).item()

        negtives[:,0]-=min_neg
        negtives[:,1:]+=max_pos

        negtives = negtives.view(-1,1)

        repeat_time = int(2*(n-1))
        postives = postives.repeat(repeat_time, 1)
        assert postives.shape[0] == negtives.shape[0]

        logit = torch.cat([postives, negtives], dim = 1)

        label = torch.zeros(logit.shape[0], dtype=torch.long).to(self.args.device)
        label = label.to(self.args.device)
        logit = logit/self.args.temperature

        return logit, label
        


    def pretrain(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start BCLR training on dataset {self.args.dataset_name} with {self.args.split} splits in random: {self.args.randoms}.")
        logging.info(f"the epoch is {self.args.epochs}, batch size is {self.args.epochs}.")
        logging.info(f"Training with no gpu: {self.args.disable_cuda}.")
        loss_list = [0]
        accuracy_list = [0]
        for epoch in trange(self.args.epochs):
            for images, _ in  train_loader:
                images = torch.cat(images, dim=0)
                image = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    # freature is (n,128) matrix
                    feature = self.model(image)
                    logits, labels = self.nce_bin_loss_MLE_constant(features=feature)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1 = accuracy(logits, labels, topk=(1,))
                    self.writer.add_scalar('pretrain loss', loss.item(), global_step= n_iter)
                    self.writer.add_scalar('pretrain acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('pretrain learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
                    for name, param in self.model.named_parameters():
                        self.writer.add_histogram(name + '_grad', param.grad, n_iter)
                        self.writer.add_histogram(name + '_data', param, n_iter)
                    loss_list.append(loss.item())
                    accuracy_list.append(top1[0])
               
                n_iter += 1    

            # wormup for the first 10 epochs
            if epoch >= 10:
                self.scheduler.step()

            logging.debug(f"Epoch: {epoch}\tLoss: {loss.item()}\tTop1 accuracy: {top1[0]}")
            


        loss_average = sum(loss_list)/len(loss_list)
        accuracy_average = sum(accuracy_list)/len(accuracy_list)
        
        logging.info("Training has finished, the average loss is {} and the average accuracy is {}.".format(loss_average, accuracy_average))
        
        # save model checkpoints
        checkpoint_name = 'CP_{}ep{:04d}bt{}.pth.tar'.format(self.args.arch, self.args.epochs,self.args.batch_size)

        save_checkpoint({'epoch': self.args.epochs,
                            'arch': self.args.arch,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),}, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")





































                    

     


