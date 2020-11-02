  
import logging
import os

import torchvision
from PIL import Image

from torch.utils.data import SubsetRandomSampler, Sampler
from torch.utils.data.dataset import ConcatDataset
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit

from augmentations import *
from common import get_logger
from imagenet import ImageNet

from augmentations import Lighting

logger = get_logger('AWSAugment')
logger.setLevel(logging.INFO)
_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def get_dataloaders(args, policy='uniform', split=0.15, split_idx=0):
    dataset = args.dataset
    batch = args.batch_size
    dataroot = args.dataroot
    if 'cifar' in dataset or 'svhn' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
    elif 'imagenet' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError('dataset=%s' % dataset)

    
    #apply current policy
    transform_train.transforms.insert(0, AWSAugmentation(policy))

 
        
    if args.cutout > 0:
        transform_train.transforms.append(CutoutDefault(args.cutout))

    if dataset == 'cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=transform_train)
        extraset = torchvision.datasets.SVHN(root=dataroot, split='extra', download=True, transform=transform_train)
        total_trainset = ConcatDataset([trainset, extraset])
        testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=transform_test)
    elif dataset == 'imagenet':
        total_trainset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), transform=transform_train)
        testset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), split='val', transform=transform_test)

        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    train_sampler = None
    if split > 0.0:
        total_idx = list(range(len(total_trainset)))
        train_idx = total_idx[:int(len(total_trainset)*(1-split))]
        valid_idx = total_idx[int(len(total_trainset)*(1-split)):]                 
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)
    else:
        valid_sampler = SubsetSampler([])

    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=True if train_sampler is None else False, num_workers=32, pin_memory=True,
        sampler=train_sampler, drop_last=True)
    validloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=False, num_workers=16, pin_memory=True,
        sampler=valid_sampler, drop_last=False)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=32, pin_memory=True,
        drop_last=False
    )
    return train_sampler, trainloader, validloader, testloader


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)