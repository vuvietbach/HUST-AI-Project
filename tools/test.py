import torch
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from FER.datasets.custom1 import FERDataset
from FER.models import CustomCNNModule
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()
def cal_res(arr, true, pred):
    true_label = list(torch.argmax(true, 1).int().cpu())
    pred_label = list(torch.argmax(pred, 1).int().cpu())
    arr['true'] = arr.get('true', [])
    arr['pred'] = arr.get('pred', [])
    
    arr['true'] += true_label
    arr['pred'] += pred_label
    return arr

def test(ckpt_path):
    # device
    device='cuda:3'
    # define model
    model = CustomCNNModule.load_from_checkpoint(ckpt_path).to(device)
    # define dataset dataloader
    test_dts = FERDataset('data/FER2013Test', 'label.csv', (64, 64))

    loader_cfg = dict(batch_size=32, num_workers=32, drop_last=False)
    dataloader = DataLoader(test_dts, **loader_cfg, shuffle=False)
    # test loop
    arr = {}
    with torch.no_grad():
        model.eval()

        for data in tqdm(dataloader):
            X, y = data
            X=X.to(device)
            y=y.to(device)
            pred = model(X)
            arr = cal_res(arr, y, pred)

    with open('result.pkl', 'wb') as f:
        pkl.dump(arr, f)
    cal_acc(arr['true'], arr['pred'])
    

def get_confusion_matrix(y_true, y_pred, cls_name, save_path):
    plt.figure()
    cls = range(len(cls_name))
    cf_mat = confusion_matrix(y_true, y_pred, labels=cls, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_mat, display_labels=cls_name)
    disp.plot()
    plt.xticks(rotation=45, ha='right')
    plt.savefig(save_path)

def cal_acc(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    acc = np.mean(y_pred==y_true)
    print(acc)

def test1():
    with open('result.pkl', 'rb') as f:
        res = pkl.load(f)
    cls_name = ['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt','unknown','NF']
    cal_acc(res['true'], res['pred'])
    
if __name__ == '__main__':
    # test1()
    test('work_dir/Best_Checkpoint-v3.ckpt')