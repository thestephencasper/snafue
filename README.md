# SNAFUE

SNAFUE = Search for Natural Adversarial Features Using Embeddings 

## Debugging Deep Neural Networks with Automated Copy/Paste Attacks

This repository accompanies the paper *Debugging Deep Neural Networks with Automated Copy/Paste Attacks* by Stephen Casper (scasper@mit.edu), Kaivu Hariharan (kaivu@mit.edu) and Dylan Hadfield-Menell. 

Preprint coming soon.

## SNAFUE with ImageNet Classifiers

SNAFUE, provides an automated method for finding targeted copy/paste attacks. This example illustrates an experiment which found that cats can make photocopiers misclassified as printers. (a) First, we create feature level adversarial patches as in Casper et al., (2022) by perturbing the latent activations of a generator. (b) We then insert natural and adversarial patches onto neural backgrounds and extract representations of them from the target network's latent activations. Finally, we select the natural patches whose latents are the most similar to the adversarial ones.

![snafue diagram](figs/diagram.pdf)

We use SNAFUE to find hundreds of weaknesses in an ImageNet classifier. See some examples below. 

![examples](figs/nat_examples.pdf)

## Getting Started

```
pip install requirements.txt

python download models.py

bash prep_models.sh

bash download_data.sh

python get_confusion_matrix.py

python get_latents.py
```

Then you can run SNAFUE

```python cp_attack.py --source=309 --targets=308```

Will run SNAFUE to find natural adversarial features that make images of bees look like flies. The results will be pickled in ```./results```. You can then load the results and show the resulting images, e.g.

```
import pickle

import matplotlib.pyplot as plt

with open('results/309_to_308.pkl', 'rb') as f:

  data = pickle.load(f)
  
print(data.keys())

plt.imshow(data['synthetic_patches[0]'])

plt.show()

plt.imshow(data['natural_patches[0]'])

plt.show()

```

Modifying the source to change details of things are run should be fairly easy. If you want to use a different network than the defail ResNet18, you'll have to modify the network loaded in ```get_confusion_matrix.py```, ```get_latents.py```, and ```cp_attack.py```. Email us if you have any questions. 
