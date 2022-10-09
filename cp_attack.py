
import sys
import argparse
import copy
import pickle
from collections import OrderedDict
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets
import torchvision.transforms as T
from fla_models.pytorch_pretrained_gans import make_gan
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

BICUBIC = InterpolationMode.BICUBIC

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--n_synthetic_total', type=int, default=15)
parser.add_argument('--n_synthetic_sample', type=int, default=10)
parser.add_argument('--n_natural_total', type=int, default=100)
parser.add_argument('--n_natural_sample', type=int, default=10)
parser.add_argument('--source', type=int, default=309)
parser.add_argument('--targets', nargs='+', type=int, default=[])
parser.add_argument('--n_targets', type=int, default=3)
parser.add_argument('--n_train_batches', type=int, default=64)
parser.add_argument('--target_network', type=str, default='resnet18')
args = parser.parse_args()
print('args parsed...')
sys.stdout.flush()

# constants
N_CLASSES = 1000
PATCH_SIDE = 64
PATCH_INSERTION_SIDE = 100
IMAGE_SIDE = 256
N_ROUND = 3
GAUSS_SIGMA = 0.12
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# transforms
resize64 = T.Resize((PATCH_SIDE, PATCH_SIDE))
resize_insertion = T.Resize((PATCH_INSERTION_SIDE, PATCH_INSERTION_SIDE))
resize256 = T.Resize((IMAGE_SIDE, IMAGE_SIDE))
resize_crop = T.Compose([T.Resize(IMAGE_SIDE), T.CenterCrop(IMAGE_SIDE)])
normalize = T.Normalize(mean=MEAN, std=STD)
unnormalize = T.Normalize(mean=-MEAN / STD, std=1 / STD)
to_tensor = T.ToTensor()


cjitter = T.ColorJitter(0.25, 0.25, 0.25, 0.05)


def custom_colorjitter(tens):
    tens = unnormalize(tens)
    tens = cjitter(tens)
    tens = normalize(tens)
    return tens


def gaussian_noise(tens, sigma=GAUSS_SIGMA):
    noise = torch.randn_like(tens) * sigma
    return tens + noise.to(device)


transforms_patch = T.Compose([custom_colorjitter, T.GaussianBlur(3, (.2, 1)), gaussian_noise,
                              T.RandomPerspective(distortion_scale=0.25, p=0.66),
                              T.RandomRotation(degrees=(-10, 10))])
row_cos = nn.CosineSimilarity(dim=1, eps=1e-6)
col_cos = nn.CosineSimilarity(dim=0, eps=1e-6)


# load certain useful data about classes, labels, etc.
with open('./data/imagenet_classes.pkl', 'rb') as f:
    class_dict = pickle.load(f)
with open('./data/confusion_matrix.pkl', 'rb') as f:
    confusion_matrix = pickle.load(f)
print('constants, transforms, ref data done...')
sys.stdout.flush()

ivs = torch.load('./data/ivs64.pth')
print('ivs loaded')
sys.stdout.flush()
bds = torch.load('./data/bds64.pth')
print('bds loaded')
sys.stdout.flush()
tin = torch.load('./data/tin64.pth')
print('tin loaded')
sys.stdout.flush()
osf = torch.load('./data/osf64.pth')
print('osf loaded')
sys.stdout.flush()

all_candidates = torch.cat([ivs, bds, tin, osf])
del ivs, bds, tin, osf
print('all patches loaded')
print(f'total candidate ims: {all_candidates.shape[0]}')
sys.stdout.flush()

val_preprocessing = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor(), normalize])
valset = datasets.ImageNet('./data/imagenet/', split='val', transform=val_preprocessing)
print('validation data loaded')
sys.stdout.flush()


class Ensemble:
    """
    Ensembles together a set of classifiers, combining them by averaging their softmax outputs.
    """

    def __init__(self, classifiers):
        self.cfs = [self.get_classifier(cf) for cf in classifiers]
        self.n_cfs = len(self.cfs)

    def get_classifier(self, name):
        if 'robust' in name:
            C = models.resnet50(pretrained=False).eval().to(device)
            model_dict = C.state_dict()
            if name == 'resnet50_robust_l2':
                load_dict = torch.load('fla_models/imagenet_l2_3_0.pt')['model']
            elif name == 'resnet50_robust_linf':
                load_dict = torch.load('fla_models/imagenet_linf_4.pt')['model']
            else:
                raise ValueError('invalid robust model name')
            new_state_dict = OrderedDict()
            for mk in model_dict.keys():
                for lk in load_dict.keys():
                    if lk[13:] == mk:
                        new_state_dict[mk] = load_dict[lk]
            C.load_state_dict(new_state_dict)
            del model_dict
            del load_dict
        else:
            lcls = locals()
            exec(f'C = models.{name}(pretrained=True).eval().to(device)', globals(), lcls)
            C = lcls['C']
        return C

    def __call__(self, inpt):
        outpts = [F.softmax(cf(inpt), 1) for cf in self.cfs]
        return sum(outpts) / self.n_cfs

# ALL_CLASSIFIERS = ['alexnet', 'resnet50', 'vgg19', 'inception_v3', 'densenet121',
#                    'resnet50_robust_l2', 'resnet50_robust_linf']
target_net = Ensemble([args.target_network])
try:
    all_latents = torch.load(f'./data/{args.target_network}_latents.pth')
except:
   raise NotImplementedError(f'latents for network {args.target_network} not found, make them with get_latents.py')
REG_CLASSIFIERS = ['resnet50_robust_l2', 'resnet50_robust_linf']
E_reg = Ensemble(REG_CLASSIFIERS)
G = make_gan(gan_type='biggan', model_name='biggan-deep-256').to(device)

nll_loss = nn.NLLLoss()  # negative log likelihood
print('models loaded')
sys.stdout.flush()


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


def tensor_to_0_1(tensor):
    """
    Shifts 0 to be at 0.5, then normalizes s.t. image falls on [0,1]
    """
    return tensor / torch.max(torch.abs(tensor)) / 2 + 0.5


def total_variation(images):
    """
    Calculates the summed L1 variation of images in tensor NCHW form
    """
    if len(images.size()) == 4:
        h_var = torch.sum(torch.abs(images[:, :, :-1, :] - images[:, :, 1:, :]))
        w_var = torch.sum(torch.abs(images[:, :, :, :-1] - images[:, :, :, 1:]))
    else:  # if 3 (CHW)
        h_var = torch.sum(torch.abs(images[:, :-1, :] - images[:, 1:, :]))
        w_var = torch.sum(torch.abs(images[:, :, :-1] - images[:, :, 1:]))
    return h_var + w_var


def entropy(sm_tensor, epsilon=1e-10):
    """
    Returns a N length vector of entropies from an NxC tensor.
    """
    log_sm_tensor = torch.log(sm_tensor + epsilon)
    h = -torch.sum(sm_tensor * log_sm_tensor, dim=1)  # formula for entropy
    return h


def custom_loss_patch_adv(output, target, patch, lam_xent=3.0, lam_tvar=1.5e-3,
                          lam_patch_xent=0.2, lam_ent=0.0, patch_bs=16):
    """
    Calculates the targeted misclassification crossentropy loss with regularization based on
    total variation, discriminator realisticness confidence, classifier patch non-target confidence,
    and classifier patch entropy.
    """
    avg_xent = nll_loss(torch.log(output), target)  # crossentropy (minimize)
    avg_tvar = total_variation(patch) / output.shape[0]  # avg total variation (minimize)
    loss = lam_xent * avg_xent + lam_tvar * avg_tvar

    if lam_patch_xent != 0 or lam_ent != 0:
        patch256 = resize256(patch)
        classifiers_out = E_reg(torch.cat([transforms_patch(patch256) for i in range(patch_bs)], axis=0)) # what the classifiers think of the patch
        patch_xent = nll_loss(torch.log(classifiers_out), target[:patch_bs])  # cross entropy loss for target (maximize)
        ent = torch.mean(entropy(classifiers_out))  # entropy for softmax outputs (minimize)
        loss -= lam_patch_xent*patch_xent
        loss += lam_ent*ent

    return loss


def insert_patch(backgrounds, patch, batch_size, prop_lower=0.2, prop_upper=0.8,
                 side_radius=10, patch_side=None, transform=True):
    """
    For universal patch attacks, this randomly tiles images and inserts patches into them.
    """
    sources = backgrounds.detach()
    orig_images = sources[:batch_size]
    images = copy.deepcopy(orig_images).to(device)
    for i in range(batch_size):
        if transform:  # randomly transform and insert
            side = np.random.randint(PATCH_SIDE, PATCH_SIDE + (2 * side_radius) + 1)
            rand_x = np.random.randint(int((IMAGE_SIDE - side) * prop_lower),
                                       int((IMAGE_SIDE - side) * prop_upper) + 1)
            rand_y = np.random.randint(int((IMAGE_SIDE - side) * prop_lower),
                                       int((IMAGE_SIDE - side) * prop_upper) + 1)
            to_insert = transforms_patch(T.functional.resize(patch.to(device), [side, side]))
            mask = to_insert != 0.0  # the mask makes any black parts of the patch not inserted
            images[i, :, rand_x: rand_x + side, rand_y: rand_y + side] *= torch.logical_not(mask)
            images[i, :, rand_x: rand_x + side, rand_y: rand_y + side] += mask * to_insert
        else:  # randomly insert
            side = PATCH_SIDE if patch_side is None else patch_side
            rand_x = np.random.randint(int((IMAGE_SIDE - side) * prop_lower),
                                       int((IMAGE_SIDE - side) * prop_upper) + 1)
            rand_y = np.random.randint(int((IMAGE_SIDE - side) * prop_lower),
                                       int((IMAGE_SIDE - side) * prop_upper) + 1)
            images[i, :, rand_x: rand_x + side, rand_y: rand_y + side] = T.functional.resize(patch.to(device), [side, side])
    return images, orig_images


def get_class_background_images(class_id):

    class_ims = torch.stack([valset[class_id * 50 + i][0] for i in range(50)])
    return class_ims


def gen_mask(layer_advs, epsilon=0.0001):
    """
    returns a mask given a set of patches that emphasizes the neurons with the least coefficient of variance
    gives a metric of which neurons are most activated by the patches
    """
    pre_mask = layer_advs.detach().cpu().reshape((len(layer_advs), -1)).numpy()
    standard_devs = np.std(pre_mask, axis=0),
    means = np.mean(pre_mask, axis=0)
    covs = np.true_divide(standard_devs, means+epsilon)
    max_cov = 1.0
    masker = torch.Tensor(np.array([0.0 if cov > max_cov else 1 - (cov / max_cov) for cov in covs[0]]))

    return masker


def get_patch_adversary(backgrounds, target_class=None, n_batches=args.n_train_batches,
                        batch_size=32, lr=0.01, latent_i=8,
                        train_noise=True, train_class_vector=True,
                        input_lr_factor=0.025, loss_hypers={}):
    """
    This function trains an adversarial patch that is targeted, universal, interpretable, and
    physically-realizable. The success rate is variable for random choices of target classes,
    so try running it multiple times.
    """
    target_tensor = torch.tensor([target_class] * batch_size, dtype=torch.long).to(device)

    # get latents from the patch generaotr
    with torch.no_grad():
        cv = torch.ones(1, 1000).to(device) / 1000
        cvp = nn.Parameter(torch.zeros_like(cv)).to(device).requires_grad_()
        nv = G.sample_latent(batch_size=1, device=device)
        nvp = nn.Parameter(torch.zeros_like(nv)).requires_grad_()
        lp = nn.Parameter(torch.zeros_like(G(nv, cv, return_latents=True)[latent_i]))
        params = [{'params': lp}]
        if train_class_vector:
            params.append({'params': cvp, 'lr': lr * input_lr_factor})
        if train_noise:
            params.append({'params': nvp, 'lr': lr * input_lr_factor})
        optimizer = optim.Adam(params, lr)

    # generate patch, insert into images, and train
    for _ in range(n_batches):
        patch = normalize(G(nv, cv, nvp, cvp, lp, insertion_layer=latent_i))
        patched_images, orig_images = insert_patch(backgrounds, patch[0], batch_size)
        predictions = target_net(patched_images)
        loss = custom_loss_patch_adv(predictions, target_tensor, patch, **loss_hypers)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # evaluate
    with torch.no_grad():
        patch = normalize(G(nv, cv, nvp, cvp, lp, insertion_layer=latent_i))
        patched_images, _ = insert_patch(backgrounds, patch[0], batch_size, transform=False, patch_side=PATCH_INSERTION_SIDE)
        adv_sm_out = target_net(patched_images)
        mean_conf = round(np.mean(np.array([float(aso[target_class]) for aso in adv_sm_out])), N_ROUND)

    return mean_conf, patch[0]


def get_patch_set(backgrounds, target_id):
    """
    gets a set of specified patches, of a given size
    """
    patches, confs = [], []
    for _ in tqdm(range(args.n_synthetic_total)):
        conf, patch = get_patch_adversary(backgrounds, target_id)
        confs.append(conf)
        patches.append(patch)
    conf_argsort = torch.argsort(torch.tensor(confs), descending=True)
    return torch.stack(patches)[conf_argsort[:args.n_synthetic_sample]]


def latent_similarity_eval(patch_latents, adv_latents, mask=None, batch_size_lat=512):
    """
    evaluates the similarity of prospective patches and generated adversaries with an optional mask argument
    """
    with torch.no_grad():
        mask_broadcast = torch.broadcast_to(mask.to(device), (batch_size_lat, adv_latents.shape[-1])) if mask is not None else 1.0
        result = []
        for adv in adv_latents:
            adv_f_mat = torch.broadcast_to(adv.to(device), (batch_size_lat, adv_latents.shape[-1]))
            if mask is not None:
                adv_f_mat = adv_f_mat * mask_broadcast
            temp = []
            for i in range(0, patch_latents.shape[0], batch_size_lat):
                try:
                    temp.append(row_cos(adv_f_mat, patch_latents[i: i + batch_size_lat].to(device)))
                except:
                    up_to = len(patch_latents[i: i + batch_size_lat])
                    temp.append(row_cos(adv_f_mat[:up_to], patch_latents[i: i + batch_size_lat].to(device)))
            img_c = torch.cat(temp)
            result.append(img_c)

    return torch.stack(result)


def eval_patch(source_ims, target_class_num, patch):
    """
    This function evaluates patches for a given class - i.e you can ask it to tell you what percent bee a given patch is
    """

    with torch.no_grad():

        backgrounds_sm_out = target_net(source_ims)
        orig_out = backgrounds_sm_out[:, target_class_num].detach().cpu()
        orig_fools = (torch.argmax(backgrounds_sm_out, dim=1) == target_class_num).sum().item()
        orig_fools = orig_fools / backgrounds_sm_out.shape[0]

        patch_stack, _ = insert_patch(source_ims, patch, len(source_ims), transform=False, patch_side=PATCH_INSERTION_SIDE)
        patch_sm_out = target_net(patch_stack)
        patch_out = patch_sm_out[:, target_class_num].detach().cpu()
        patch_fools = (torch.argmax(patch_sm_out, dim=1) == target_class_num).sum().item()
        patch_fools = patch_fools / patch_stack.shape[0]

    return torch.mean(patch_out - orig_out).item(), (patch_fools - orig_fools)


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def run_attack(source_class, target_class, mask=True):
    """
    parameters: source class, target class, optional focus classifier
    returns: adv_patches, nat_img_patches, nat_img_labels, captions, mean_fooling_confidence
    """

    background_images = get_class_background_images(source_class).to(device)
    adv_patches = resize64(get_patch_set(background_images, target_class))
    adv_mean_fooling_conf_increase, adv_mean_fooling_rate_increase = [], []
    for patch in adv_patches:
        fooling_conf_increase, fooling_rate_increase = eval_patch(background_images, target_class, patch)
        adv_mean_fooling_conf_increase.append(round(fooling_conf_increase, N_ROUND))
        adv_mean_fooling_rate_increase.append(round(fooling_rate_increase, N_ROUND))

    try:
        target_net.cfs[0].avgpool.register_forward_hook(get_activation('avgpool'))
    except:
        raise NotImplementedError('Edit the above line if using a network other than a resnet')
    classifier_inpt = resize_insertion(adv_patches).to(device)
    _ = target_net.cfs[0](classifier_inpt)
    try:
        adv_latents = torch.squeeze(activation['avgpool'])
    except:
        raise NotImplementedError('Edit the above line if using a network other than a resnet')

    mask_tensor = gen_mask(adv_latents.detach()) if mask else None
    nat_img_simil = latent_similarity_eval(all_latents, adv_latents, mask=mask_tensor)

    # this evaluates by the mean square cos similarity
    max_res = np.array(torch.mean((nat_img_simil**2).detach(), 0).cpu())

    # This loop gets the natural images that are the most similar to the advs but that aren't of the target class
    max_res_argsort = np.flip(np.argsort(max_res))
    natural_patches, natural_patch_idxs = [], []
    for id in max_res_argsort:
        im = resize_insertion(all_candidates[[id]]).to(device)
        with torch.no_grad():
            label = torch.argmax(target_net(im), dim=1)[0].item()
            if label != target_class:
                natural_patch_idxs.append(id)
                natural_patches.append(all_candidates[id])
        if len(natural_patches) >= args.n_natural_total:
            break

    natural_patches = torch.stack(natural_patches)

    nat_mean_fooling_conf_increase, nat_mean_fooling_rate_increase = [], []
    for patch in natural_patches:
        fooling_conf_increase, fooling_rate_increase = eval_patch(background_images, target_class, patch)
        nat_mean_fooling_conf_increase.append(round(fooling_conf_increase, N_ROUND))
        nat_mean_fooling_rate_increase.append(round(fooling_rate_increase, N_ROUND))

    nat_conf_argsort = np.flip(np.argsort(np.array(nat_mean_fooling_conf_increase)))
    natural_patches = [natural_patches[ni] for ni in nat_conf_argsort[:args.n_natural_sample]]
    natural_patch_idxs = [natural_patch_idxs[ni] for ni in nat_conf_argsort[:args.n_natural_sample]]
    nat_mean_fooling_conf_increase = [nat_mean_fooling_conf_increase[ni] for ni in nat_conf_argsort[:args.n_natural_sample]]
    nat_mean_fooling_rate_increase = [nat_mean_fooling_rate_increase[ni] for ni in nat_conf_argsort[:args.n_natural_sample]]

    return (adv_patches, natural_patches, natural_patch_idxs,
            adv_mean_fooling_conf_increase, nat_mean_fooling_conf_increase,
            adv_mean_fooling_rate_increase, nat_mean_fooling_rate_increase)


if __name__ == '__main__':

    print('\nStart :)')
    sys.stdout.flush()
    t0 = time()

    if args.targets:
        targets = args.targets
    else:
        confusion_row = confusion_matrix[args.source]
        targets = np.flip(np.argsort(confusion_row)[-(args.n_targets + 1):])

    for target_class in targets:
        if target_class == args.source:
            continue

        (adv_patches, natural_patches, natural_patch_idxs,
         adv_mean_fooling_conf_increase, nat_mean_fooling_conf_increase,
         adv_mean_fooling_rate_increase, nat_mean_fooling_rate_increase) = run_attack(args.source, target_class)
        print('source_class', args.source, class_dict[args.source])
        print('target_class', target_class, class_dict[target_class])
        print('adv_mean_fooling_conf_increase', adv_mean_fooling_conf_increase)
        print('nat_mean_fooling_conf_increase', nat_mean_fooling_conf_increase)
        print('adv_mean_fooling_rate_increase', adv_mean_fooling_rate_increase)
        print('nat_mean_fooling_rate_increase', nat_mean_fooling_rate_increase)

        save_dict = {'source_class': args.source,
                     'target_class': target_class,
                     'synthetic_patches': adv_patches,
                     'natural_patches': natural_patches,
                     'natural_patch_idxs': natural_patch_idxs,
                     'synthetic_mean_fooling_conf_increase': adv_mean_fooling_conf_increase,
                     'nat_mean_fooling_conf_increase': nat_mean_fooling_conf_increase,
                     'synthetic_mean_fooling_rate_increase': adv_mean_fooling_rate_increase,
                     'nat_mean_fooling_rate_increase': nat_mean_fooling_rate_increase}
        with open(f'results/{args.source}_to_{target_class}.pkl', 'wb') as f:
            pickle.dump(save_dict, f)

        print()
        t1 = time()
        print(f'time: {round((t1 - t0) / 60)}m')

    print('Done :)')

