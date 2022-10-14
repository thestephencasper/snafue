# setup step 5

import numpy as np
import torch
import argparse
import torchvision
from torchvision import models
from torchvision import datasets
from torchvision import transforms as T
from tqdm import tqdm
import pickle

assert torch.cuda.is_available()
device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('--target_network', type=str, default='resnet18')
parser.add_argument('--weights_path', type=str, default='')
args = parser.parse_args()


MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
BATCH_SIZE = 128  # 256
TRAIN_EPS = 2
IM_SIDE = 256
normalize = T.Normalize(mean=MEAN, std=STD)

val_preprocessing = T.Compose([T.Resize(256), T.CenterCrop(IM_SIDE), T.ToTensor(), normalize])
valset = datasets.ImageNet('./data/imagenet/', split='val', transform=val_preprocessing)
val_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)


def get_matrix(model, loader):

    cm = torch.zeros((1000, 1000)).to(device)
    model.eval()

    with torch.no_grad():
        for X, y in tqdm(loader):

            X = X.to(device)
            y_hat = model(X)

            for i in range(len(y)):
                cm[y[i]] += y_hat[i] / 50

    return cm.detach().cpu().numpy()


if __name__ == '__main__':

    if args.weights_path:
        lcls = locals()
        exec(f'model = models.{args.target_network}(pretrained=False).eval().to(device)', globals(), lcls)
        model = lcls['model']
        model.load_state_dict(torch.load(args.weights_path))
    else:
        lcls = locals()
        exec(f'model = models.{args.target_network}(pretrained=True).eval().to(device)', globals(), lcls)
        model = lcls['model']

    cm = get_matrix(model, val_loader)
    with open('./data/confusion_matrix.pkl', 'wb') as f:
        pickle.dump(cm, f)
