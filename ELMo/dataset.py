import torch
from torch.utils.data import Dataset

class CharacterDataset(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index]), torch.tensor(self.label[index])
