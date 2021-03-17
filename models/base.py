


import torch.nn as nn
import abc

class ADType(abc.ABC): ...

class AdversarialDefensiveModel(ADType, nn.Module):
    """
    Define some basic properties.
    """
    def __init__(self):
        super(AdversarialDefensiveModel, self).__init__()
        # Some model's outputs for training(evaluating) 
        # and attacking are different.
        self.attacking = False

        
    def attack(self, mode=True):
        # enter attacking mode
        self.attacking = mode
        for module in self.children():
            if isinstance(module, ADType):
                module.attack(mode)




if __name__ == "__main__":
    
    model = AdversarialDefensiveModel()
    model.child1 = AdversarialDefensiveModel()
    model.child2 = AdversarialDefensiveModel()

    print(model.attack)
    model.attack()
    for m in model.children():
        print(m.attacking)

    model.defense()
    for m in model.children():
        print(m.attacking)

