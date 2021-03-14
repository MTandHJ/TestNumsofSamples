
import torch
from torch.utils.data.sampler import Sampler



class ReplaceableSubsetSampler(Sampler):

    def __init__(self, source_data, nums):
        self.source_data = source_data
        self.nums = nums if nums is not None else len(self.source_data)
        assert len(source_data) >= self.nums, "The given number is bigger than the owned."

    def __iter__(self):
        n = len(self.source_data)
        idx = torch.randint(high=self.nums, size=(n, ), dtype=torch.int64)
        rdx = torch.randperm(n)
        return iter(idx[rdx].tolist())
    
    def __len__(self):
        return len(self.source_data)


