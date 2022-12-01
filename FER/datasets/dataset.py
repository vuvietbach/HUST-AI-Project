import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip
class FERDataset(Dataset):
    def __init__(self, csv_file, data_dir=None, mode='train'):
        self.df = pd.read_csv(csv_file)
        self.data_dir = '.' if data_dir is None else data_dir
        self.transforms = Compose([
            RandomHorizontalFlip(),
            Lambda(lambda x : np.array(x)),
            Lambda(lambda x : x / 255.0),
            Lambda(lambda x:np.expand_dims(x, axis=0)),
            Lambda(lambda x:x.astype(np.float32))]
        ) if mode == 'train' else Compose([
            Lambda(lambda x : np.array(x)),
            Lambda(lambda x : x / 255.0),
            Lambda(lambda x:np.expand_dims(x, axis=0)),
            Lambda(lambda x:x.astype(np.float32))]
        )
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = osp.join(self.data_dir, self.df.loc[idx, 'image'])
        img = Image.open(img_path)
        img = self.transforms(img)
        
        label = self.df.loc[idx, 'label'].item()
        return {'X':img, 'y':label}

if __name__ == '__main__':
    dts = FERDataset('csv_file/test.csv', mode='test')
    print(dts[0])
