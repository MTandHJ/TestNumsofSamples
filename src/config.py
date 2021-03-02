







# Here are some basic settings.
# It could be overwritten if you want to specify
# some configs. However, please check the correspoding
# codes in loadopts.py.



import torchvision.transforms as T
import random
from PIL import ImageFilter
from .dict2obj import Config



class _GaussBlur:

    def __init__(self, sigma=(.1, 2.)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x



ROOT = "../data"
INFO_PATH = "./infos/{method}/{dataset}-{model}/{description}"
LOG_PATH = "./logs/{method}/{dataset}-{model}/{description}"
TIMEFMT = "%m%d%H"

TRANSFORMS = {
    "mnist": {
        'default': T.ToTensor()
    },
    "cifar10": {
        'default': T.Compose((
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        )),
        'simclr': T.Compose((
            T.RandomResizedCrop(32, scale=(0.2, 1.0)),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([_GaussBlur()], p=0.5),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ))
    },
    "cifar100": {
        'default': T.Compose((
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        )),
        'simclr': T.Compose((
            T.RandomResizedCrop(32, scale=(0.2, 1.0)),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([_GaussBlur()], p=0.5),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ))
    }
}

VALIDER = {
    "mnist": (Config(attack_type="pgd-linf", stepsize=0.033333, steps=100), 0.3),
    "cifar10": (Config(attack_type="pgd-linf", stepsize=0.1, steps=20), 8/255),
    "cifar100": (Config(attack_type="pgd-linf", stepsize=0.1, steps=20), 8/255)
}

# env settings
NUM_WORKERS = 3
PIN_MEMORY = True

# basic properties of inputs
BOUNDS = (0, 1)
MEANS = {
    "mnist": None,
    "cifar10": [0.4914, 0.4824, 0.4467],
    "cifar100": [0.5071, 0.4867, 0.4408]
}

STDS = {
    "mnist": None,
    "cifar10": [0.2471, 0.2435, 0.2617],
    "cifar100": [0.2675, 0.2565, 0.2761]
}

# the settings of optimizers of which lr could be pointed
# additionally.
OPTIMS = {
    "sgd": Config(lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=False),
    "adam": Config(lr=0.01, betas=(0.9, 0.999), weight_decay=0.)
}


# the learning schedular can be added here
LEARNING_POLICY = {
   "default": (
        "MultiStepLR",
        Config(
            milestones=[100, 105],
            gamma=0.1
        ),
        "Default leaning policy will be applied: " \
        "decay the learning rate at 100 and 105 epochs by a factor 10."
    ),
    "STD": (
        "MultiStepLR",
        Config(
            milestones=[82, 123],
            gamma=0.1
        ),
        "STD leaning policy will be applied: " \
        "decay the learning rate at 82 and 123 epochs by a factor 10."
    ),
    "AT":(
        "MultiStepLR",
        Config(
            milestones=[102, 154],
            gamma=0.1
        ),
        "AT learning policy, an official config, " \
        "decays the learning rate at 102 and 154 epochs by a factor 10 for total 200 epochs."
    ),
    "TRADES":(
        "MultiStepLR",
        Config(
            milestones=[75, 90, 100],
            gamma=0.1
        ),
        "TRADES learning policy, an official config, " \
        "decays the learning rate at 75 epochs by factor 10 for total 76 epochs."
    ),
    "cosine":(   
        "CosineAnnealingLR",   
        Config(          
            T_max=200,
            eta_min=0.,
            last_epoch=-1,
        ),
        "cosine learning policy: T_max == epochs - 1"
    )
}






