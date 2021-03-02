
import torch
import torchvision
import torchvision.transforms as T
import foolbox as fb
from tqdm import tqdm


from .base import Adversary
from .dict2obj import Config
from .config import *



class ModelNotDefineError(Exception): pass
class LossNotDefineError(Exception): pass
class OptimNotIncludeError(Exception): pass
class AttackNotIncludeError(Exception): pass
class DatasetNotIncludeError(Exception): pass


# return the num_classes of corresponding data set
def get_num_classes(dataset_type: str):
    if dataset_type in ('mnist', 'cifar10'):
        return 10
    elif dataset_type in ('cifar100', ):
        return 100
    else:
        raise DatasetNotIncludeError("Dataset {0} is not included." \
                        "Refer to the following: {1}".format(dataset_type, _dataset.__doc__))

def get_num_samples(dataset_type: str):
    if dataset_type in ("cifar10", "cifar100"):
        return 5000
    elif dataset_type in ("mnist", ):
        return 6000
    else:
        raise DatasetNotIncludeError("Dataset {0} is not included." \
                        "Refer to the following: {1}".format(dataset_type, _dataset.__doc__))


def load_model(model_type: str):
    """
    mnist: the model designed for MNIST dataset
    cifar: the model designed for CIFAR dataset
    resnet20|32|44|110|1202
    resnet18|34|50|101|50_32x4d
    wrn-28-10: depth-28, width-10
    wrn-34-10: depth-34, width-10
    wrn-34-20: depth-34, width-20
    """
    resnets = ['resnet20', 'resnet32', 'resnet44', 
                'resnet56', 'resnet110', 'resnet1202']
    srns = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnext50_32x4d']
    wrns = ['wrn-28-10', 'wrn-34-10', 'wrn-34-20']

    if model_type == "mnist":
        from models.mnist import MNIST
        model = MNIST
    elif model_type == "cifar":
        from models.cifar import CIFAR
        model = CIFAR
    elif model_type in resnets:
        import models.resnet as resnet
        model = getattr(resnet, model_type)
    elif model_type in srns:
        import models.cifar_resnet as srn
        model = getattr(srn, model_type)
    elif model_type in wrns:
        import models.wide_resnet as wrn
        model = getattr(wrn, model_type)
    else:
        raise ModelNotDefineError(f"model {model_type} is not defined.\n" \
                f"Refer to the following: {load_model.__doc__}\n")
    return model


def load_loss_func(loss_type: str):
    """
    cross_entropy: the softmax cross entropy loss
    kl_loss: kl divergence
    """
    if loss_type == "cross_entropy":
        from .loss_zoo import cross_entropy
        loss_func = cross_entropy
    elif loss_type == "contrastive":
        from .loss_zoo import contrastive_loss
        loss_func = contrastive_loss
    elif loss_type == "kl_loss":
        from .loss_zoo import kl_divergence
        loss_func = kl_divergence
    elif loss_type == "margin_loss":
        from .loss_zoo import margin_loss
        loss_func = margin_loss
    else:
        raise LossNotDefineError(f"Loss {loss_type} is not defined.\n" \
                    f"Refer to the following: {load_loss_func.__doc__}")
    return loss_func


class _Normalize:

    def __init__(self, mean=None, std=None):
        self.set_normalizer(mean, std)

    def set_normalizer(self, mean, std):
        if mean is None or std is None:
            self.flag = False
            return 0
        self.flag = True
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        self.nat_normalize = T.Normalize(
            mean=mean, std=std
        )
        self.inv_normalize = T.Normalize(
            mean=-mean/std, std=1/std
        )

    def _normalize(self, imgs, inv):
        if not self.flag:
            return imgs
        if inv:
            normalizer = self.inv_normalize
        else:
            normalizer = self.nat_normalize
        new_imgs = [normalizer(img) for img in imgs]
        return torch.stack(new_imgs)

    def __call__(self, imgs, inv=False):
        # normalizer will set device automatically.
        return self._normalize(imgs, inv)


def _get_normalizer(dataset_type: str):
    mean = MEANS[dataset_type]
    std = STDS[dataset_type]
    return _Normalize(mean, std)


def _get_transform(dataset_type: str, transform: str, train=True):
    if train:
        return TRANSFORMS[dataset_type][transform]
    else:
        return T.ToTensor()


def _dataset(dataset_type: str, transform: str,  train=True):
    """
    Dataset:
    mnist: MNIST
    cifar10: CIFAR-10
    cifar100: CIFAR-100
    Transform:
    default: the default transform for each data set
    simclr: the transform introduced in SimCLR
    """
    try:
        transform = _get_transform(dataset_type, transform, train)
    except KeyError:
        raise DatasetNotIncludeError(f"Dataset {dataset_type} or transform {transform} is not included.\n" \
                        f"Refer to the following: {_dataset.__doc__}")

    if dataset_type == "mnist":
        dataset = torchvision.datasets.MNIST(
            root=ROOT, train=train, download=False,
            transform=transform
        )
    elif dataset_type == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=ROOT, train=train, download=False,
            transform=transform
        )
    elif dataset_type == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root=ROOT, train=train, download=False,
            transform=transform
        )
        
    return dataset


def load_normalizer(dataset_type: str):
    normalizer = _get_normalizer(dataset_type)
    return normalizer


def load_dataset(dataset_type: str, transform='default', train=True, nums=None):
    dataset = _dataset(dataset_type, transform, train)
    from .datasets import SingleSet
    dataset = SingleSet(
        dataset=dataset, transform=lambda x: x,
        nums=nums
    )
    return dataset


class _TQDMDataLoader(torch.utils.data.DataLoader):
    def __iter__(self):
        return iter(
            tqdm(
                super(_TQDMDataLoader, self).__iter__(), 
                leave=False, desc="վ'ᴗ' ի-"
            )
        )


def load_dataloader(dataset, batch_size: int, train=True, show_progress=False):
    dataloader = _TQDMDataLoader if show_progress else torch.utils.data.DataLoader
    if train:
        dataloader = dataloader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY)
    else:
        dataloader = dataloader(dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY)

    return dataloader


def load_optimizer(
    model: torch.nn.Module, 
    optim_type: str, *,
    lr=0.1, momentum=0.9,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
    nesterov=False,
    **kwargs
):
    """
    sgd: SGD
    adam: Adam
    """
    try:
        cfg = OPTIMS[optim_type]
    except KeyError:
        raise OptimNotIncludeError(f"Optim {optim_type} is not included.\n" \
                        f"Refer to the following: {load_optimizer.__doc__}")
    
    kwargs.update(lr=lr, momentum=momentum, betas=betas, 
                weight_decay=weight_decay, nesterov=nesterov)
    
    cfg.update(**kwargs) # update the kwargs needed automatically
    print(optim_type, cfg)
    if optim_type == "sgd":
        optim = torch.optim.SGD(model.parameters(), **cfg)
    elif optim_type == "adam":
        optim = torch.optim.Adam(model.parameters(), **cfg)

    return optim


def load_learning_policy(
    optimizer: torch.optim.Optimizer,
    learning_policy_type: str,
    **kwargs
):
    """
    default: (100, 105), 110 epochs
    STD: (82, 123), 200 epochs
    AT: (102, 154), 200 epochs
    TRADES: (75, 90, 100), 76 epochs
    cosine: CosineAnnealingLR, kwargs: T_max, eta_min, last_epoch
    """
    try:
        learning_policy_ = LEARNING_POLICY[learning_policy_type]
    except KeyError:
        raise NotImplementedError(f"Learning_policy {learning_policy_type} is not defined.\n" \
            f"Refer to the following: {load_learning_policy.__doc__}")

    lp_type = learning_policy_[0]
    lp_cfg = learning_policy_[1]
    lp_description = learning_policy_[2]
    lp_cfg.update(**kwargs) # update the kwargs needed automatically
    print(lp_type, lp_cfg)
    print(lp_description)
    learning_policy = getattr(
        torch.optim.lr_scheduler, 
        lp_type
    )(optimizer, **lp_cfg)
    
    return learning_policy


def _get_preprocessing(dataset_type: str):
    preprocessing = None
    if dataset_type in ("cifar10", "cifar100"):
        mean = MEANS[dataset_type]
        std = STDS[dataset_type]
        preprocessing = dict(
            mean=mean,
            std=std,
            axis=-3
        )
    return preprocessing


def _attack(attack_type: str, stepsize: float, steps: int):
    """
    pgd-linf: \ell_{\infty} rel_stepsize=stepsize, steps=steps;
    pgd-l1: \ell_1 version;
    pgd-l2: \ell_2 version;
    pgd-kl: pgd-linf with KL divergence loss;
    fgsm: no hyper-parameters;
    cw-l2: stepsize=stepsize, steps=steps;
    ead: initial_stepsize=stepsize, steps=steps;
    sparse-l1: rel_stepsize=stepsize, steps=steps;
    deepfool-linf: \ell_{\infty} version, overshoot=stepsize, steps=steps;
    deepfool-l2: \ell_2 version;
    bba-inf: \ell_{infty} version, lr=stepsize, steps=steps, overshott=1.1;
    bba-l1: \ell_1 version;
    bba-l2: \ell_2 version
    """
    if attack_type == "pgd-linf":
        attack = fb.attacks.LinfPGD(
            rel_stepsize=stepsize,
            steps=steps
        )
    elif attack_type == "pgd-l2":
        attack = fb.attacks.L2PGD(
            rel_stepsize=stepsize,
            steps=steps
        )
    elif attack_type == "pgd-l1":
        attack = fb.attacks.L1PGD(
            rel_stepsize=stepsize,
            steps=steps
        )
    elif attack_type == "pgd-kl":
        from .attacks import LinfPGDKLDiv
        attack = LinfPGDKLDiv(
            rel_stepsize=stepsize,
            steps=steps
        )
    elif attack_type == "fgsm":
        attack = fb.attacks.LinfFastGradientAttack(
            random_start=False
        )
    elif attack_type == "cw-l2":
        attack = fb.attacks.L2CarliniWagnerAttack(
            stepsize=stepsize,
            steps=steps
        )
    elif attack_type == "ead":
        attack = fb.attacks.EADAttack(
            initial_stepsize=stepsize,
            steps=steps
        )
    elif attack_type == "sparse-l1":
        attack = fb.attacks.SparseL1DescentAttack(
            rel_stepsize=stepsize,
            steps=steps
        )
    elif attack_type == "deepfool-linf":
        attack = fb.attacks.LinfDeepFoolAttack(
            overshoot=stepsize,
            steps=steps
        )
    elif attack_type == "deepfool-l2":
        attack = fb.attacks.L2DeepFoolAttack(
            overshoot=stepsize,
            steps=steps
        )
    elif attack_type == "bba-linf":
        attack = fb.attacks.LinfinityBrendelBethgeAttack(
            lr=stepsize,
            steps=steps
        )
    elif attack_type == "bba-l2":
        attack = fb.attacks.L2BrendelBethgeAttack(
            lr=stepsize,
            steps=steps
        )
    elif attack_type == "bba-l1":
        attack = fb.attacks.L1BrendelBethgeAttack(
            lr=stepsize,
            steps=steps
        )
    else:
        raise AttackNotIncludeError(f"Attack {attack_type} is not included.\n" \
                    f"Refer to the following: {_attack.__doc__}")
    return attack


def load_attacks(attack_type: str, dataset_type: str, stepsize: float, steps: int):
    attack = _attack(attack_type, stepsize, steps)
    preprocessing = _get_preprocessing(dataset_type)
    bounds = BOUNDS
    return attack, bounds, preprocessing


def load_valider(model: torch.nn.Module, device, dataset_type: str):
    cfg, epsilon = VALIDER[dataset_type]
    attack, bounds, preprocessing = load_attacks(dataset_type=dataset_type, **cfg)
    valider = Adversary(
        model, attack, device, 
        bounds, preprocessing, epsilon
    )
    return valider


def generate_path(method: str, dataset_type: str, model:str,  description: str):
    info_path = INFO_PATH.format(
        method=method,
        dataset=dataset_type,
        model=model,
        description=description
    )
    log_path = LOG_PATH.format(
        method=method,
        dataset=dataset_type,
        model=model,
        description=description
    )
    return info_path, log_path

