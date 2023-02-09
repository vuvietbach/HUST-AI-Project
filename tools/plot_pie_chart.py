import torch
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
CLASSES = ['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt','unknown','NF']

def plot_pie_chart(path, plot_title, plot_save_path):
    with open(path,'r') as f:
        reader = csv.reader(f)
        data = [row[2:] for row in reader]
    data = [np.argmax(np.asarray(prop)).item() for prop in data]
    data = [CLASSES[i] for i in data]
    df = pd.Series(data)
    value_counts = df.value_counts()
    plt.figure()
    value_counts.plot.pie(autopct='%1.1f%%', figsize=(7, 7), title=plot_title)
    plt.savefig(plot_save_path,  bbox_inches='tight')
    
if __name__ == '__main__':
    plot_pie_chart('data/FER2013Train/label.csv', 'train', 'test_images/train_distributions.png')
    plot_pie_chart('data/FER2013Test/label.csv', 'test', 'test_images/test_distributions.png')

