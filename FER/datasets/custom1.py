import numpy as np
import os.path as osp
import csv

from PIL import Image
from torch.utils.data import Dataset, DataLoader

class FERDataset(Dataset):
    def __init__(self, base_folder, csv_file, target_size):
        self.base_folder = base_folder
        
        csv_file = osp.join(base_folder, csv_file)
        with open(csv_file) as f:
            reader = csv.reader(f)
            data = [[row[0], row[2:]] for row in reader]
        self.data = self.process_data(data)
    
        self.target_size = target_size

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # read image
        img_path = osp.join(self.base_folder, self.data[idx][0])
        img = Image.open(img_path)
        img = self.process_input(img)
        
        # read label propabilities
        target = self.data[idx][1]

        return img, target
    
    def process_input(self, input):
        input = input.resize(self.target_size)
        input = np.array(input, dtype=np.float32)
        input = input / 255.0
        input = np.expand_dims(input, 0)
        return input

    def process_data(self, datas):
        # normalize
        for data in datas:
            prop = np.array(data[1], dtype=np.float32)
            prop = (prop + 0.001)/np.sum(np.array(prop)+0.001)
            data[1] = prop
        
        return datas

if __name__ == '__main__':
    dts = FERDataset('data/FER2013Test', 'label.csv', (64, 64))
    cfg = dict(num_workers=32, batch_size=4, drop_last=False, shuffle=True)
    loader = DataLoader(dts, **cfg)
    for data in loader:
        x, y = data
        print(x.shape)
        print(y)
        break
