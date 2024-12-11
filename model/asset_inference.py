import torch
from asset_data_preprocess import DataPreprocessor
import pandas as pd
from torch.utils.data import DataLoader

batch_size = 4096
# preprocessor = DataPreprocessor('train_B2C.csv', 'test_B2C.csv', batch_size=batch_size)
# train_loader, val_loader, test_loader = preprocessor.process()
train_dataset = pd.read_csv('localdata/data_without_pctv.csv')
val_dataset = pd.read_csv('localdata/data_with_pctv.csv')
test_dataset = pd.read_csv('localdata/test.csv')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Inference:
    def __init__(self, device='cpu', name='model_name'):
        print(device)
        
        self.model = torch.load(name, weights_only=False)
        self.device = device
        self.name = name

    def run(self):
        self.model.eval()
        data = {}

        logit = self.model(data)
        threshold = self.model.threshold
        subscription_score = logit / threshold
        
        if logit > threshold:
            print(f"subscription score is {subscription_score}. Yes subscription")
        else:
            print(f"subscription score is {subscription_score}. No subscription")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
find_threshold = Inference(device=device, name="reloop_with_pctv.pth")
find_threshold.run()