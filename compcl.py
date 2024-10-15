"""
This scripts contains the comparison methods in our work
"""

import logging
import os
from tqdm import trange
import numpy as np

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from utils import save_config_file, accuracy, save_checkpoint


class COMPCL(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        termpath = "{}_ep{:04d}bt{}".format(self.args.arch, self.args.epochs,self.args.batch_size)
        log_dir = os.path.join("PretrainModel",self.args.method ,self.args.dataset_name, termpath)
        self.writer = SummaryWriter(log_dir=log_dir)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'train.log'), level= logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        # self.criterion = torch.nn.BCEWithLogitsLoss().to(self.args.device)

    def SimCL_loss(self, features, **kwargs):
        # SimCL loss
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        loss = self.criterion(logits, labels)
        return loss
    

    def tri_SimCL_loss(self, features, temperature: float = 0.1, **kwargs):
        """
        Tri-factor Contrastive Learning loss function from:https://openreview.net/pdf?id=BQA7wR2KBF
        and code: https://github.com/PKU-ML/Tri-factor-Contrastive-Learning
        """
        # the loss function L_{dec}(f) 
        scale_param = kwargs['scale_param']
        N, D = features.size()
        features = F.normalize(features, dim=-1).to(self.args.device)
        corr = features.T @ features / N
        diag = torch.eye(D, device=corr.device)
        cdif = (corr - diag).pow(2)
        dec_loss = cdif.mean()

        # the loss function L_{tri}(f,S) where the S = scale_param
        features1, features2 = torch.chunk(features, 2)
        # scale_param = torch.nn.Parameter(torch.randn(128)*0.1)
        scale = torch.nn.functional.softplus(scale_param).unsqueeze(0).to(self.args.device)
        features1 = features1*scale
        features = torch.cat([features1, features2])

        indexes  = torch.tensor([i for i in range(self.args.batch_size)])
        indexes  = indexes.repeat(2).unsqueeze(0).to(self.args.device)
        # #positives
        pos_mask = indexes.t() == indexes

        similarity = torch.einsum("if, jf -> ij", features, features)

        # get logits and labels
        mask = torch.eye(pos_mask.shape[0], dtype=torch.bool).to(self.args.device)
        pos_mask = pos_mask[~mask].view(pos_mask.shape[0], -1)
        sim_mask = similarity[~mask].view(similarity.shape[0], -1)

        pos_sim = sim_mask[pos_mask.bool()].view(sim_mask.shape[0], -1)
        neg_sim = sim_mask[~pos_mask.bool()].view(sim_mask.shape[0], -1)
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        logits /= temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        sim_loss = self.criterion(logits, labels)

        loss = sim_loss + dec_loss

        return loss
    
    def epsilon_SupInfoNCE_loss(self, features, base_temperature=0.07, **kwargs):
        """
        The unbiased supervised contrastive loss: https://openreview.net/pdf?id=Ph5cJSfD2XN
        reference code: https://github.com/EIDOSLAB/unbiased-contrastive-learning
        """
        device = features.device

        omega = kwargs['omega']
        temperature = kwargs['temperature']

        n = features.shape[0]/2
        mask =  torch.cat([torch.arange(n) for i in range(2)] ,dim=0)
        mask = (mask.unsqueeze(0) == mask .unsqueeze(1)).float().bool()
        mask = mask.to(device)

        # Inverse of torch-eye to remove self-contrast (diagonal)
        inv_diagonal = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(features.shape[0], device=device).view(-1, 1),
            0
        ).bool()
        # mask now contains the anchor-positive pairs
        # excluding <self,self> on the diagonal
        positive_mask = mask * inv_diagonal

        # compute similarity
        features = F.normalize(features, dim=-1)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        """ 
        $ \log \left( \sum_i \left( \frac{\exp(s_i^+)}{\frac{1}{P} \sum_i \exp(s_i^+ - \epsilon)) +  \sum_j \exp(s_j^-)} \right) \right) $
        """
        alignment = torch.log((torch.exp(logits) * positive_mask).sum(1, keepdim=True)) 
        
        uniformity = torch.exp(logits) * inv_diagonal 
        uniformity = ((omega * uniformity * positive_mask) / \
                        torch.max(positive_mask.sum(1, keepdim=True), 
                                  torch.ones_like(positive_mask.sum(1, keepdim=True)))) + \
                     (uniformity * (~positive_mask) * inv_diagonal)
        uniformity = torch.log(uniformity.sum(1, keepdim=True))

        log_prob = alignment - uniformity
        loss = -(temperature / base_temperature) * log_prob
        
        return loss.mean()


    def debiase_cl_loss(self, features, **kwargs ):
        """
        The debiased cotrastive learning:https://proceedings.neurips.cc/paper/2020/file/63c3ddcc7b23daa1e42dc41f9a44a873-Paper.pdf
        reference code: https://github.com/chingyaoc/DCL
        """
        temperature = kwargs['temperature']
        tau_plus = kwargs['tau_plus']
        features = F.normalize(features, dim=-1)
        sim = torch.exp(torch.matmul(features, features.T) / temperature)

        diag = torch.eye(features.shape[0], dtype=torch.bool)
        mask = torch.cat([torch.arange(self.args.batch_size) for i in range(2)], dim=0)
        mask = (mask.unsqueeze(0) == mask.unsqueeze(1)).float()
        mask = mask[~diag].view(mask.shape[0], -1).bool()
        sim  = sim[~diag].view(mask.shape[0], -1)

        neg = sim[~mask].view(2 * self.args.batch_size, -1)
        pos = sim[mask].view(2 * self.args.batch_size, -1).squeeze()

        # estimator g()
        N = self.args.batch_size * 2 - 2
        Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))

        # contrastive loss
        loss = (-torch.log(pos / (pos + Ng))).mean()

        return loss


    def pretrain(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start COMPCL training on dataset {self.args.dataset_name} with {self.args.split} splits in random: {self.args.randoms}.")
        logging.info(f"the epoch is {self.args.epochs}, batch size is {self.args.batch_size}.")
        logging.info(f"Training with no gpu: {self.args.disable_cuda}.")
        loss_list = [0]
        # accuracy_list = [0]

        # prepare the parameters for different methods
        if self.args.method == "sim_cl":
            loss_func = self.SimCL_loss
            kwargs = {}
        elif self.args.method == "tri_sim_cl":
            loss_func = self.tri_SimCL_loss
            scale_param = torch.nn.Parameter(torch.randn(128)*0.1 )
            kwargs = {"scale_param": scale_param}
        elif self.args.method == "e_infnce_loss":   
            loss_func = self.epsilon_SupInfoNCE_loss
            temperature = 0.1
            epsilon = 0.0001
            omega = np.exp(-epsilon)
            kwargs = {'omega': omega, 'temperature': temperature}
        elif self.args.method == "debiase_loss":     
            loss_func = self.debiase_cl_loss
            temperature = 0.1
            tau_plus = 0.1
            kwargs = {'tau_plus': tau_plus, 'temperature': temperature }
        else:
            raise ValueError("Such {} method is not  available!".format(self.args.method))


        for epoch in trange(self.args.epochs):
            for images, _ in  train_loader:
                images = torch.cat(images, dim=0)
                image = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    # freature is (n,128) matrix
                    feature = self.model(image)
                    loss = loss_func(feature, **kwargs)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('pretrain loss', loss.item(), global_step= n_iter)
                    self.writer.add_scalar('pretrain learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
                    for name, param in self.model.named_parameters():
                        self.writer.add_histogram(name + '_grad', param.grad, n_iter)
                        self.writer.add_histogram(name + '_data', param, n_iter)
                    loss_list.append(loss.item())
               
                n_iter += 1    

            # wormup for the first 10 epochs
            if epoch >= 10:
                self.scheduler.step()

            logging.debug(f"Epoch: {epoch}\tLoss: {loss.item()}")
            


        loss_average = sum(loss_list)/len(loss_list)
        
        logging.info("Training has finished, the average loss is {}.".format(loss_average))
        
        # save model checkpoints
        checkpoint_name = 'CP_{}ep{:04d}bt{}.pth.tar'.format(self.args.arch, self.args.epochs,self.args.batch_size)

        save_checkpoint({'epoch': self.args.epochs,
                            'arch': self.args.arch,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),}, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")





































                    

     


