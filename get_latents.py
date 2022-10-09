
# setup step 6

from tqdm import tqdm
import numpy as np
import torch
import argparse
import torchvision.models as models
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC

assert torch.cuda.is_available()
device = 'cuda'


parser = argparse.ArgumentParser()
parser.add_argument('--target_network', type=str, default='resnet18')
args = parser.parse_args()


# constants
N_CLASSES = 1000
PATCH_SIDE = 64
PATCH_INSERTION_SIDE = 100
IMAGE_SIDE = 256
N_ROUND = 4
GAUSS_SIGMA = 0.12
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


# transforms
resize_insertion = T.Resize((PATCH_INSERTION_SIDE, PATCH_INSERTION_SIDE))
normalize = T.Normalize(mean=MEAN, std=STD)
unnormalize = T.Normalize(mean=-MEAN / STD, std=1 / STD)
to_tensor = T.ToTensor()


def numpy_image_to_tensor(array, normalize_img=True):
    """
    Takes a 3-channel numpy image to a tensor that can be fed into networks'
    """
    array = np.transpose(array, (2, 0, 1))
    maxval = 1.0 if np.max(array) <= 1 else 255.0
    # print(maxval)
    # print(np.max(array))
    n_array = array / maxval
    f_array = np.clip(n_array, 0, 1)
    tensor = torch.tensor(f_array, device=device, dtype=torch.float).unsqueeze(0)
    if tensor.shape[1] == 4:
        tensor = tensor[:, :3, :, :]
    return normalize(tensor) if normalize_img else tensor


ivs = torch.load('./data/ivs64.pth')
bds = torch.load('./data/bds64.pth')
tin = torch.load('./data/tin64.pth')
osf = torch.load('./data/osf64.pth')
all_candidates = torch.cat([ivs, bds, tin, osf])
del ivs, bds, tin, osf
print('data loaded :)')


def tensor_to_numpy_image(tensor, unnormalize_img=True):
    """
    Takes a tensor and turns it into an imshowable np.ndarray
    """
    image = tensor
    if unnormalize_img:
        image = unnormalize(image)
    image = image.detach().cpu().numpy()
    image = np.squeeze(image)
    image = np.transpose(image, axes=(1, 2, 0))
    image = np.clip(image, 0, 1)
    return image


def tensor_to_numpy_image_batch(tensor, unnormalize_img=True):
    """
    Takes a tensor and turns it into an imshowable np.ndarray
    """
    image = tensor
    if unnormalize_img:
        image = unnormalize(image)
    image = image.detach().cpu().numpy()
    image = np.squeeze(image)
    image = np.transpose(image, axes=(0, 2, 3, 1))
    image = np.clip(image, 0, 1)
    return image


def tensor_to_0_1(tensor):
    """
    Shifts 0 to be at 0.5, then normalizes s.t. image falls on [0,1]
    """
    return tensor / torch.max(torch.abs(tensor)) / 2 + 0.5


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, ims):
        self.ims = ims

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        return self.ims[idx]


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def get_latents(model, model_name, dataloader):

    try:
        model.avgpool.register_forward_hook(get_activation('avgpool'))
    except:
        raise NotImplementedError('Edit the above line if using a network other than a resnet')

    with torch.no_grad():

        latents = []
        for X in tqdm(dataloader):
            X_insertion = resize_insertion(X).to(device)
            _ = model(X_insertion)
            try:
                latents.append(torch.squeeze(activation['avgpool']))
            except:
                raise NotImplementedError('Edit the above line if using a network other than a resnet')

        latents = torch.cat(latents, dim=0)

        print(latents.shape, torch.min(latents), torch.max(latents))

        torch.save(latents, f'./data/{model_name}_latents.pth')


if __name__ == '__main__':

    dset = SimpleDataset(all_candidates)
    dataloader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=False)

    lcls = locals()
    exec(f'model = models.{args.target_network}(pretrained=True).eval().to(device)', globals(), lcls)
    model = lcls['model']

    get_latents(model, args.target_network, dataloader)

