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
from utils import calc_test_loss, ReLoopLoss, CustomDataset
from asset_data_preprocess import DataPreprocessor

batch_size = 4096

# preprocessor = DataPreprocessor('train_B2C.csv', 'test_B2C.csv', batch_size=batch_size)
# no_use, train_loader, test_loader = preprocessor.process()

train_dataset = pd.read_csv('localdata/data_with_pctv.csv')
test_dataset = pd.read_csv('localdata/test.csv')

train_loader = DataLoader(CustomDataset(train_dataset), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(CustomDataset(test_dataset), batch_size=batch_size, shuffle=False)

class Reloop:
    def __init__(self, device='cpu', name='model_name', load_name='prev_model_name'):
        print(device)
        
        self.prev_model = torch.load(load_name, weights_only=False)
        self.model = torch.load(load_name, weights_only=False)

        self.prev_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = ReLoopLoss(alpha=0.9)
        self.prev_criterion = nn.BCELoss()
        self.epochs = 50
        self.device = device
        self.name = name

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            train_loss = 0.0
            for data, label in train_loader:
                data, label = data.to(self.device).long(), label.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(data).squeeze()
                prev_outputs = self.prev_model(data).squeeze()
                loss = self.criterion(label.float(), outputs, prev_outputs)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # Validation loop
            self.model.eval()

            train_loss /= len(train_loader)

            prediction_loss, f1_score = calc_test_loss(self.model, test_loader, self.prev_criterion, self.device)

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Test F1 score : {f1_score}")

    def save(self):
        torch.save(self.model, self.name)
    

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

reloop = Reloop(device=device, load_name = "fmlp_without_pctv.pth", name="reloop_with_pctv.pth")
reloop.train()
reloop.save()

prediction_loss, f1_score = calc_test_loss(reloop.model, test_loader, reloop.prev_criterion, device)

print(f"Test Loss: {prediction_loss:.4f}, F1 Score : {f1_score}")

        
