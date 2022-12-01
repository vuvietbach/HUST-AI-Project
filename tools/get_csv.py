import pandas as pd
import glob
import os.path as osp
import random
categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
cat_id = {cat:id for cat, id in zip(categories, range(7))}

def get_csv(crawl_dir, save_path):
    data = []
    for id, category in enumerate(categories):
        path1 = osp.join(crawl_dir, category)
        files = list(glob.glob(f'{path1}/*.jpg'))
        data.extend([(id, file) for file in files])
    df = pd.DataFrame(data, columns=['label', 'image'])
    df.to_csv(save_path, index=False)

def get_datalist(crawl_dir, data):
    for id, category in enumerate(categories):
        path1 = osp.join(crawl_dir, category)
        files = list(glob.glob(f'{path1}/*.jpg'))
        data[id] = data.get(id, [])
        data[id].extend([(id, file) for file in files])
    return data
def split_data(data, ratios):
    random.shuffle(data)
    total_len = len(data)
    len_lst = [int(total_len*ratio) for ratio in ratios]
    index = 0
    res = []
    for len_dta in len_lst:
        res.append(data[index:index+len_dta])
        index += len_dta
    if index < total_len:
        res.append(data[index:])
    return res
    
if __name__ == '__main__':
    df = pd.read_csv('csv_file/val.csv')
    data = {}
    data = get_datalist('images/train', data)
    data = get_datalist('images/validation', data)
    data1 = [[], [], []]
    for cat_data in data.values():
        data2 = split_data(cat_data, [0.7, 0.2])
        for data3, data4 in zip(data1, data2):
            data3.extend(data4)
    
    
    save_csv = [f'csv_file/{file}' for file in ['train.csv', 'val.csv', 'test.csv']]
    for file, dfdata in zip(save_csv, data1):
        df = pd.DataFrame(dfdata, columns=['label', 'image'])
        df.to_csv(file, index=False)

