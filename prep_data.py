# setup step 4

import os
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision
from torchvision import transforms as T
from torchvision import models, datasets
import numpy as np
# from netdissect.broden import BrodenDataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def generate_dataloader(data, transform):
    if data is None:
        return None
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=T.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)
    # Set options for device
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}
    # Wrap image dataset (defined above) in dataloader
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, **kwargs)
    return dataloader


MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
preprocess_transform = T.Compose([
    T.Resize(64),  # Resize images to 256 x 256
    T.CenterCrop(64),  # Center crop image
    T.ToTensor(),  # Converting cropped images to tensors
    T.Normalize(mean=MEAN, std=STD)
])
preprocess_transform_imagenet_valset = T.Compose([
    T.Resize(256),  # Resize images to 256 x 256
    T.CenterCrop(64),  # Center crop image
    T.ToTensor(),  # Converting cropped images to tensors
    T.Normalize(mean=MEAN, std=STD)
])


# imagenet validation set
ivs = datasets.ImageNet('./data/imagenet/', split='val', transform=preprocess_transform_imagenet_valset)
ivs_all_x = [ivs[i][0] for i in range(len(ivs))]
ivs_all_x = torch.stack(ivs_all_x)
print(ivs_all_x.shape)
torch.save(ivs_all_x, 'data/ivs64.pth')
del ivs, ivs_all_x
print('ivs done')


# broden
bds_img_dir = './data/broden1_224/images'
bds_loader = generate_dataloader(bds_img_dir, transform=preprocess_transform)
bds_all_x = []
for x, _ in bds_loader:
    bds_all_x.append(x)

bds_all_x = torch.cat(bds_all_x, dim=0)
print(bds_all_x.shape)
torch.save(bds_all_x, './data/bds64.pth')
del bds_all_x, bds_loader
print('bds done')


# tiny imagenet
tin_img_dir = './data/tiny-imagenet-200'
tin_loader = generate_dataloader(tin_img_dir, transform=preprocess_transform)
tin_loader2 = generate_dataloader(tin_img_dir, transform=preprocess_transform)
tin_all_x = []
for x, _ in tin_loader:
    tin_all_x.append(x)

tin_all_x = torch.cat(tin_all_x, dim=0)
print(tin_all_x.shape)
torch.save(tin_all_x, './data/tin64.pth')
del tin_all_x, tin_loader, tin_loader2
print('tin done')


# open surfaces dataset
osf_img_dir = './data/minc-2500/images'
osf_loader = generate_dataloader(osf_img_dir, transform=preprocess_transform)
osf_all_x = []
for x, _ in osf_loader:
    osf_all_x.append(x)

osf_all_x = torch.cat(osf_all_x, dim=0)
print(osf_all_x.shape)
torch.save(osf_all_x, './data/osf64.pth')
del osf_all_x, osf_loader
print('osf done')
