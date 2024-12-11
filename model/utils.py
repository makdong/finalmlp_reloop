from sklearn.metrics import precision_score, recall_score
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import nn

def f1_score_metric(y_true, y_pred):
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

def calc_test_loss(model, test_loader, criterion, device):
    y_pred = []
    y_true = []
    prediction_loss = 0.0

    model.eval()
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device).long(), label.to(device)

            outputs = model(data).squeeze()
            loss = criterion(outputs, label.float())
            prediction_loss += loss.item()
            if device == 'cpu':
                y_pred.extend(outputs)
                y_true.extend(label)
            else:
                y_pred.extend(outputs.cpu().tolist())
                y_true.extend(label.cpu().tolist())

    predicted = [1 if y > 0.5 else 0 for y in y_pred]

    prediction_loss /= len(test_loader)
    f1 = f1_score_metric(y_true, predicted)

    return prediction_loss, f1

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.features = dataframe.drop(columns=['y']).values  # 입력 데이터
        self.labels = dataframe['y'].values  # 레이블

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 특정 샘플의 feature와 label 반환
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label

class ReLoopLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(ReLoopLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()
    
    def self_correction_loss(self, y_true, y_pred, y_last):
        positive_loss = y_true * torch.clamp(y_last - y_pred, min=0)
        negative_loss = (1 - y_true) * torch.clamp(y_pred - y_last, min=0)
        return positive_loss + negative_loss

    def forward(self, y_true, y_pred, y_last):
        sc_loss = self.self_correction_loss(y_true, y_pred, y_last)
        ce_loss = self.bce_loss(y_pred, y_true.float())

        return self.alpha * sc_loss.mean() + (1 - self.alpha) * ce_loss