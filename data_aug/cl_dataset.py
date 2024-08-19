# import numpy as np
import random
import torch
import numpy as np
from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from torch.utils.data import Subset
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
 

class CLDataset:
    def __init__(self, root_folder, Totransform = True):
        self.root_folder = root_folder
        self.Totransform = Totransform

    # @staticmethod
    def get_CL_pipeline_transform(self, size, s = 1):
        if self.Totransform:
            color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
            data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomApply([color_jitter], p=0.8),
                                                transforms.RandomGrayscale(p=0.2),
                                                GaussianBlur(kernel_size=int(0.1 * size)),
                                                transforms.ToTensor()])
            # data_transforms = ContrastiveLearningViewGenerator(data_transform_s, 2)
        else:
            data_transforms = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor()])
        return data_transforms    

     
    def get_dataset(self, name):
        valid_traindatsets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True, transform=ContrastiveLearningViewGenerator(self.get_CL_pipeline_transform(32),2), download= True) if self.Totransform 
                              else datasets.CIFAR10(self.root_folder, train=True, transform=self.get_CL_pipeline_transform(32), download= True),
                         
                         'cifar100':lambda: datasets.CIFAR100(self.root_folder, train=True, transform=ContrastiveLearningViewGenerator(self.get_CL_pipeline_transform(32),2), download= True) if self.Totransform 
                            else datasets.CIFAR100(self.root_folder, train=True, transform=self.get_CL_pipeline_transform(32), download= True),
                         
                         "stl10":lambda: datasets.STL10(self.root_folder, split="unlabeled", transform=ContrastiveLearningViewGenerator(self.get_CL_pipeline_transform(96),2), download= True) if self.Totransform
                            else datasets.STL10(self.root_folder, split="train", transform=self.get_CL_pipeline_transform(96), download= True),
                        #  'caltech101':lambda :datasets.Caltech101(self.root_folder, train=True, transform=self.get_CL_pipeline_transform(32), download= True),

                         'food101':lambda : datasets.Food101(self.root_folder, split="train", transform=ContrastiveLearningViewGenerator(self.get_CL_pipeline_transform(200),2), download= True) if self.Totransform
                            else datasets.Food101(self.root_folder, split="train", transform=self.get_CL_pipeline_transform(200), download= True),
                        #  'stanfordcars':lambda :datasets.StanfordCars(self.root_folder, split="train", transform=self.get_CL_pipeline_transform(32), download= True),

                         'dtd':lambda : datasets.DTD(self.root_folder, split="train", transform=ContrastiveLearningViewGenerator(self.get_CL_pipeline_transform(200),2), download= True) if self.Totransform
                            else datasets.DTD(self.root_folder, split="train", transform=self.get_CL_pipeline_transform(200), download= True),

                         'oxford3':lambda : datasets.OxfordIIITPet(self.root_folder, split='trainval', transform=ContrastiveLearningViewGenerator(self.get_CL_pipeline_transform(200),2), download= True) if self.Totransform
                            else datasets.OxfordIIITPet(self.root_folder, split='trainval', transform=self.get_CL_pipeline_transform(200), download= True)
                         }
        

        valid_testdatsets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=False, transform=self.get_CL_pipeline_transform(32), download= True),
                         
                         'cifar100':lambda: datasets.CIFAR100(self.root_folder, train=False, transform=self.get_CL_pipeline_transform(32), download= True),
                         
                         "stl10":lambda: datasets.STL10(self.root_folder, split='test', transform=self.get_CL_pipeline_transform(96), download= True),

                        #  'caltech101':lambda :datasets.Caltech101(self.root_folder, train=False, transform=self.get_CL_pipeline_transform(32), download= True),

                         'food101':lambda :datasets.Food101(self.root_folder, split='test', transform=self.get_CL_pipeline_transform(200), download= True),

                        #  'stanfordcars':lambda :datasets.StanfordCars(self.root_folder, split='test', transform=self.get_CL_pipeline_transform(32), download= True),

                         'dtd':lambda :datasets.DTD(self.root_folder, split='test', transform=self.get_CL_pipeline_transform(200), download= True),

                         'oxford3':lambda :datasets.OxfordIIITPet(self.root_folder, split='test', transform=self.get_CL_pipeline_transform(200), download= True)

                         }
       
        try:
            dataset_trainfn = valid_traindatsets[name]
            dataset_testfn = valid_testdatsets[name]
        except KeyError:
            raise InvalidDatasetSelection() 
        else:
            return dataset_trainfn(), dataset_testfn()  

class Customsubset(Subset):
    """To keep the attribute of origin data """
    def __init__(self, dataset, indices, pos_ind, neg_ind):
        super().__init__(dataset, indices)
        self.pos_ind = pos_ind
        self.neg_ind = neg_ind
        self.dataset = [dataset[i] for  i in indices]
        self.targets = [0 for i in self.pos_ind] + [1 for j in self.neg_ind]  


    def __getitem__(self, indices):
        dataset, targets = self.dataset[indices], self.targets[indices]
        return dataset, targets    
    
# class BiCustomsubset:
#     def __init__(self, pos_ind, neg_ind, Trmtrainset, Trmtestset):
#         self.pos_ind = pos_ind
#         self.neg_ind = neg_ind
#         self.Trmtrainset = Trmtrainset
#         self.Trmtestset = Trmtestset 

#     def Binarydatagen(pos_ind, neg_ind, trianset, testset):
#         Pos_train_ind = np.where(np.array(trianset.targets) == pos_ind)[0]
#         Neg__train_ind = np.where(np.array(trianset.targets) == neg_ind)[0]
#         Binary_train_ind = np.concatenate((Pos_train_ind, Neg__train_ind))
#         Binary_tain = Customsubset(trianset, Binary_train_ind,Pos_train_ind, Neg__train_ind)

#         Pos_test_ind = np.where(np.array(testset.targets) == pos_ind)[0]
#         Neg__test_ind = np.where(np.array(testset.targets) == neg_ind)[0]
#         Binary_test_ind = np.concatenate((Pos_test_ind, Neg__test_ind))
#         Binary_test = Customsubset(testset, Binary_test_ind, Pos_test_ind, Neg__test_ind)

#         return Binary_tain, Binary_test     

        