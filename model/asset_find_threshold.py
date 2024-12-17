import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from final_mlp import FinalMLP
from utils import f1_score_metric, ReLoopLoss, CustomDataset
from asset_data_preprocess import DataPreprocessor
from itertools import chain

batch_size = 4096

# preprocessor = DataPreprocessor('train_B2C.csv', 'test_B2C.csv', batch_size=batch_size)
# train_loader, val_loader, test_loader = preprocessor.process()

train_dataset = pd.read_csv('localdata/data_without_pctv.csv')
val_dataset = pd.read_csv('localdata/data_with_pctv.csv')
test_dataset = pd.read_csv('localdata/test.csv')

train_loader = DataLoader(CustomDataset(train_dataset), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(CustomDataset(val_dataset), batch_size=batch_size)
test_loader = DataLoader(CustomDataset(test_dataset), batch_size=batch_size, shuffle=False)

class FindThreshold:
    def __init__(self, device='cpu', name='model_name'):
        print(device)
        
        self.model = torch.load(name, weights_only=False)
        self.device = device
        self.name = name

    def run(self):
        y_pred = []
        y_true = []
        
        final_threshold = 0.5
        final_f1 = 0
        
        for data, label in chain(train_loader, val_loader):
            data, label = data.to(self.device).long(), label.to(self.device)
            outputs = self.model(data).squeeze()

            if device == 'cpu':
                y_pred.extend(outputs)
                y_true.extend(label)
            else:
                y_pred.extend(outputs.cpu().tolist())
                y_true.extend(label.cpu().tolist())

        for i in tqdm(range(100)):
            threshold = i * 0.01

            predicted = [1 if y > threshold else 0 for y in y_pred]
            f1 = f1_score_metric(y_true, predicted)

            if final_f1 < f1:
                final_f1 = f1
                final_threshold = threshold

        self.model.threshold = final_threshold
        torch.save(self.model, self.name)

        y_pred = []
        y_true = []

        for data, label in test_loader:
            data, label = data.to(self.device).long(), label.to(self.device)
            outputs = self.model(data).squeeze()

            if device == 'cpu':
                y_pred.extend(outputs)
                y_true.extend(label)
            else:
                y_pred.extend(outputs.cpu().tolist())
                y_true.extend(label.cpu().tolist())

        predicted = [1 if y > self.model.threshold else 0 for y in y_pred]
        test_f1 = f1_score_metric(y_true, predicted)


        print(f"Final threshold: {final_threshold} | Final F1 score : {final_f1} | Test F1 score : {test_f1}")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
find_threshold = FindThreshold(device=device, name="fmlp_without_pctv.pth")
# find_threshold = FindThreshold(device=device, name="reloop_with_pctv.pth")
find_threshold.run()

        
