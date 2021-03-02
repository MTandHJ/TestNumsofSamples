
import torch


class SingleSet(torch.utils.data.Dataset):

    def __init__(self, dataset, transform, *, nums=None):
        """
        dataset: ...
        transform: the transformation
        nums: the number of source images for training
        """
        self.data = dataset
        self.transform = transform
        self.counts = len(self.data)
        self.nums = self.counts if nums is None else nums
        assert self.nums < self.counts, \
            f"The nums of needed {self.nums} less than maximum {self.counts}."
        print(f"Set the number of source images for training: {nums}")

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        img, label = self.data[index]
        return self.transform(img), label
