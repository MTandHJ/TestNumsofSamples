


import torch
import torch.nn as nn
import foolbox as fb
import eagerpy as ep
from models.base import AdversarialDefensiveModel
from .criteria import LogitsAllFalse
from .utils import AverageMeter, ProgressMeter
from .loss_zoo import cross_entropy, kl_divergence



def enter_attack_exit(func):
    def wrapper(attacker, *args, **kwargs):
        attacker.model.attack(True)
        results = func(attacker, *args, **kwargs)
        attacker.model.attack(False)
        return results
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


class Coach:
    """
    Coach is used to train models.
    model: ...
    device: ...
    loss_func: ...
    normalizer: an explicit transformer for inputs
    optimizer: sgd, adam, ...
    learning_policy: for learning rate decay
    """
    def __init__(
        self, model, device,
        loss_func, normalizer,
        optimizer, learning_policy
    ):
        self.model = model
        self.device = device
        self.loss_func = loss_func
        self.normalizer = normalizer
        self.optimizer = optimizer
        self.learning_policy = learning_policy
        self.loss = AverageMeter("Loss")
        self.acc = AverageMeter("Acc")
        self.progress = ProgressMeter(self.loss, self.acc)
        
    def save(self, path):
        torch.save(self.model.state_dict(), path + "/paras.pt")

    def train(self, trainloader, *, epoch=8888):
        self.progress.step() # reset the meter
        self.model.train()
        for inputs, labels in trainloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.model.train() # make sure in training mode
            outs = self.model(self.normalizer(inputs))
            loss = self.loss_func(outs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            accuracy_count = (outs.argmax(-1) == labels).sum().item()
            self.loss.update(loss.item(), inputs.size(0))
            self.acc.update(accuracy_count, inputs.size(0), mode="sum")

        self.progress.display(epoch=epoch) 
        self.learning_policy.step() # update the learning rate
        return self.loss.avg

    def adv_train(self, trainloader, attacker, *, epoch=8888):
        assert isinstance(attacker, Adversary)
        self.progress.step() # reset the meter
        for inputs, labels in trainloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            _, clipped, _ = attacker(inputs, labels)
            
            self.model.train()
            outs = self.model(self.normalizer(clipped))
            loss = self.loss_func(outs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            accuracy_count = (outs.argmax(-1) == labels).sum().item()
            self.loss.update(loss.item(), inputs.size(0))
            self.acc.update(accuracy_count, inputs.size(0), mode="sum")

        self.progress.display(epoch=epoch)
        self.learning_policy.step() # update the learning rate
        return self.loss.avg

    def trades(self, trainloader, attacker, *, leverage=6., epoch=8888):
        self.progress.step()
        for inputs, labels in trainloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                self.model.eval()
                logits = self.model(self.normalizer(inputs)).detach()
            criterion = LogitsAllFalse(logits) # perturbed by kl loss
            _, inputs_adv, _ = attacker(inputs, criterion)
            
            self.model.train()
            logits_clean = self.model(self.normalizer(inputs))
            logits_adv = self.model(self.normalizer(inputs_adv))
            loss_clean = cross_entropy(logits_clean, labels)
            loss_adv = kl_divergence(logits_adv, logits_clean)
            loss = loss_clean + leverage * loss_adv

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc_count = (logits_adv.argmax(-1) == labels).sum().item()
            self.loss.update(loss.item(), inputs.size(0), mode="mean")
            self.acc.update(acc_count, inputs.size(0), mode="sum")

        self.progress.display(epoch=epoch)
        self.learning_policy.step()

        return self.loss.avg



class FBDefense:
    def __init__(self, model, device, bounds, preprocessing):
        self.rmodel = fb.PyTorchModel(
            model,
            bounds=bounds,
            preprocessing=preprocessing,
            device=device            
        )

        self.model = model
    
    def train(self, mode=True):
        self.model.train(mode=mode)

    def eval(self):
        self.train(mode=False)

    def query(self, inputs):
        return self.rmodel(inputs)

    def __call__(self, inputs):
        return self.query(inputs)


class Adversary:
    """
    Adversary is mainly based on foolbox, especially pytorchmodel.
    model: Make sure that the model's output is the logits or the attack is adapted.
    attacker: the attack implemented by foolbox or a similar one
    device: ...
    bounds: typically [0, 1]
    preprocessing: including mean, std, which is similar to normalizer
    criterion: typically given the labels and consequently it is Misclassification, 
            other critera could be given to carry target attack or black attack.
    """
    def __init__(
        self, model, attacker, device,
        bounds, preprocessing, epsilon
    ):
        model.eval()
        self.fmodel = fb.PyTorchModel(
            model,
            bounds=bounds,
            preprocessing=preprocessing,
            device=device
        )
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.attacker = attacker 

    @enter_attack_exit
    def attack(self, inputs, criterion, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        self.model.eval() # make sure in evaluating mode ...
        return self.attacker(self.fmodel, inputs, criterion, epsilons=epsilon)

    @torch.no_grad()
    def accuracy(self, inputs, labels):
        inputs_, labels_ = ep.astensors(inputs, labels)
        del inputs, labels

        self.model.eval() # make sure in evaluating mode ...
        predictions = self.fmodel(inputs_).argmax(axis=-1)
        accuracy = (predictions == labels_)
        return accuracy.sum().item()

    def success(self, inputs, criterion, epsilon=None):
        _, _, is_adv = self.attack(inputs, criterion, epsilon)
        return is_adv.sum().item()

    def evaluate(self, dataloader, epsilon=None):
        datasize = len(dataloader.dataset)
        running_accuracy = 0
        running_success = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            running_accuracy += self.accuracy(inputs, labels)
            running_success += self.success(inputs, labels, epsilon)
        return running_accuracy / datasize, running_success / datasize

    def __call__(self, inputs, criterion, epsilon=None):
        return self.attack(inputs, criterion, epsilon)



