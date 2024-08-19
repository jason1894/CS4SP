import os
import shutil

import torch
import yaml

"""Save the pretrianed model to desk"""
def save_checkpoint(state, is_best, filename= 'checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'modal_best.pth.tar')


def save_config_file(model_chekpoints_folder, args):
    if not os.path.exists(model_chekpoints_folder):
        os.makedirs(model_chekpoints_folder)
        with open(os.path.join(model_chekpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k = 2"""
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
