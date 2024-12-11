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
from utils import calc_test_loss, CustomDataset
from asset_data_preprocess import DataPreprocessor

batch_size = 4096

preprocessor = DataPreprocessor('train_B2C.csv', 'test_B2C.csv', batch_size=batch_size)
train_loader, val_loader, test_loader = preprocessor.process()

train_dataset = pd.read_csv('localdata/data_without_pctv.csv')
val_dataset = pd.read_csv('localdata/data_with_pctv.csv')
test_dataset = pd.read_csv('localdata/test.csv')

train_loader = DataLoader(CustomDataset(train_dataset), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(CustomDataset(val_dataset), batch_size=batch_size)
test_loader = DataLoader(CustomDataset(test_dataset), batch_size=batch_size, shuffle=False)

class FMLP:
    def __init__(self, num_features=32, num_embedding=512, dim_embedding=16, dropout=0.2, device='cpu', name='model_name'):
        print(device)

        self.model = FinalMLP(
                        dim_input=num_features,
                        num_embedding=num_embedding,
                        dim_embedding=dim_embedding,
                        dropout=dropout,
                    ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
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
                loss = self.criterion(outputs, label.float())
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # Validation loop
            self.model.eval()

            val_loss = 0.0
            with torch.no_grad():
                for data, label in val_loader:
                    data, label = data.to(self.device).long(), label.to(self.device)

                    outputs = self.model(data).squeeze()
                    loss = self.criterion(outputs, label.float())
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            prediction_loss, f1_score = calc_test_loss(self.model, test_loader, self.criterion, self.device)

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test F1 score : {f1_score}")

    def save(self):
        torch.save(self.model, self.name)
    
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

fmlp = FMLP(device=device, name="fmlp_without_pctv.pth")
fmlp.train()
fmlp.save()

prediction_loss, f1_score = calc_test_loss(fmlp.model, test_loader, fmlp.criterion, device)

print(f"Test Loss: {prediction_loss:.4f}, F1 Score : {f1_score}")

        
