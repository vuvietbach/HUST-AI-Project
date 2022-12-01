from mobilenetv3 import MobilenetModule
from torch.utils.data import DataLoader
import torch
import pandas as pd
from dataset import FERDataset
def test(ckpt_path):
    # device
    device='cuda:3'
    # define model
    model = MobilenetModule.load_from_checkpoint(ckpt_path).to(device)
    # define dataset dataloader
    dataset = FERDataset('csv_file/train.csv', 'image')
    dataloader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=False)
    # test loop
    with torch.no_grad():
        model.eval()
        # logging 
        values = []
        total_correct = 0
        for data in dataloader:
            data['X']=data['X'].to(device)
            data['y']=data['y'].to(device)
            pred = model(data['X'])
            label = torch.argmax(pred, dim=1)
            total_correct += torch.sum(label == data['y'])

            for i in range(label.shape[0]):
                values.append(label[i].item())

        print(f'accuracy : {total_correct / len(dataset)}')
        series = pd.Series(values, name='predict_label')
        series.to_csv('csv_file/result.csv', index=False)
if __name__ == '__main__':
    test('work_dir/Best_Checkpoint-v5.ckpt')