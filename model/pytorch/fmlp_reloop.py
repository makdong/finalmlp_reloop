import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Sequential):
    def __init__(self, dim_in, num_hidden, dim_hidden, dim_out=None, batch_norm=True, dropout=0.0):
        layers = []
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(dim_in, dim_hidden))

            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_hidden))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dim_in = dim_hidden

        if dim_out:
            layers.append(nn.Linear(dim_in, dim_out))
        else:
            layers.append(nn.Linear(dim_in, dim_hidden))

        super().__init__(*layers)


class FeatureSelection(nn.Module):
    def __init__(self, dim_input, num_hidden=1, dim_hidden=64, dropout=0.0):
        super().__init__()

        self.gate_1 = MLP(
            dim_in=dim_input,
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=dim_input,
            dropout=dropout,
            batch_norm=False,
        )

        self.gate_2 = MLP(
            dim_in=dim_input,
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=dim_input,
            dropout=dropout,
            batch_norm=False,
        )

    def forward(self, inputs):
        gate_score_1 = self.gate_1(inputs)  # (1, dim_input)
        out_1 = 2.0 * F.sigmoid(gate_score_1) * inputs  # (bs, dim_input)

        gate_score_2 = self.gate_2(inputs)  # (1, dim_input)
        out_2 = 2.0 * F.sigmoid(gate_score_2) * inputs  # (bs, dim_input)

        return out_1, out_2  # (bs, dim_input), (bs, dim_input)


class Aggregation(nn.Module):
    def __init__(self, dim_inputs_1, dim_inputs_2, num_heads=1):
        super().__init__()

        self.num_heads = num_heads
        self.dim_head_1 = dim_inputs_1 // num_heads
        self.dim_head_2 = dim_inputs_2 // num_heads

        self.w_1 = nn.Parameter(torch.empty(self.dim_head_1, num_heads, 1))
        self.w_2 = nn.Parameter(torch.empty(self.dim_head_2, num_heads, 1))
        self.w_12 = nn.Parameter(torch.empty(num_heads, self.dim_head_1, self.dim_head_2, 1))
        self.bias = nn.Parameter(torch.ones(1, num_heads, 1))

        self._reset_weights()

    def _reset_weights(self):
        nn.init.xavier_uniform_(self.w_1)
        nn.init.xavier_uniform_(self.w_2)
        nn.init.xavier_uniform_(self.w_12)

    def forward(self, inputs_1, inputs_2):
        # bilinear aggregation of the two latent representations
        # y = b + w_1.T o_1 + w_2.T o_2 + o_1.T W_3 o_2
        inputs_1 = torch.reshape(inputs_1, (-1, self.num_heads, self.dim_head_1))  # (bs, num_heads, dim_head_1)
        inputs_2 = torch.reshape(inputs_2, (-1, self.num_heads, self.dim_head_2))  # (bs, num_heads, dim_head_2)

        first_order = torch.einsum("bhi,iho->bho", inputs_1, self.w_1)  # (bs, num_heads, 1)
        first_order += torch.einsum("bhi,iho->bho", inputs_2, self.w_2)  # (bs, num_heads, 1)
        second_order = torch.einsum("bhi,hijo,bhj->bho", inputs_1, self.w_12, inputs_2)  # (bs, num_heads, 1)

        out = torch.sum(first_order + second_order + self.bias, dim=1)  # (bs, 1)

        return out


class FinalMLP(nn.Module):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding=32,
        dim_hidden_fs=64,
        num_hidden_1=2,
        dim_hidden_1=64,
        num_hidden_2=2,
        dim_hidden_2=64,
        num_heads=1,
        dropout=0.0,
    ):
        super().__init__()

        self.dim_input = dim_input
        self.dim_embedding = dim_embedding

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=dim_embedding)

        # feature selection layer that projects a learnable vector to the flatened embedded feature space
        self.feature_selection = FeatureSelection(
            dim_input=dim_input * dim_embedding,
            dim_hidden=dim_hidden_fs,
            dropout=dropout,
        )

        # branch 1
        self.interaction_1 = MLP(
            dim_in=dim_input * dim_embedding,
            num_hidden=num_hidden_1,
            dim_hidden=dim_hidden_1,
            dropout=dropout,
        )
        # branch 2
        self.interaction_2 = MLP(
            dim_in=dim_input * dim_embedding,
            num_hidden=num_hidden_2,
            dim_hidden=dim_hidden_2,
            dropout=dropout,
        )

        # final aggregation layer
        self.aggregation = Aggregation(
            dim_inputs_1=dim_hidden_1,
            dim_inputs_2=dim_hidden_2,
            num_heads=num_heads,
        )

    def forward(self, inputs):
        embeddings = self.embedding(inputs)  # (bs, num_emb, dim_emb)
        embeddings = torch.reshape(embeddings, (-1, self.dim_input * self.dim_embedding))  # (bs, num_emb * dim_emb)

        # weight features of the two streams using a gating mechanism
        emb_1, emb_2 = self.feature_selection(embeddings)  # (bs, num_emb * dim_emb), (bs, num_emb * dim_emb)

        # get interactions from the two branches
        # (bs, dim_hidden_1), (bs, dim_hidden_1)
        latent_1, latent_2 = self.interaction_1(emb_1), self.interaction_2(emb_2)

        # merge the representations using an aggregation scheme
        logits = self.aggregation(latent_1, latent_2)  # (bs, 1)
        outputs = F.sigmoid(logits)  # (bs, 1)

        return outputs

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class PandasDataset(Dataset):
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

# 데이터 불러오기
train_data = pd.read_csv('train_B2C.csv')
test_data = pd.read_csv('test_B2C.csv')

# 필요 없는 열 제거
train_data = train_data.drop(columns=['train/test', 'cust_id', 'base_dt'])
test_data = test_data.drop(columns=['train/test', 'cust_id', 'base_dt'])

# 결측치 처리
rcnt_columns = ['rcnt_sub_0', 'rcnt_sub_1', 'rcnt_sub_2', 'rcnt_sub_3', 'rcnt_sub_4',
                'rcnt_buy_0', 'rcnt_buy_1', 'rcnt_buy_2', 'rcnt_buy_3', 'rcnt_buy_4']
for col in rcnt_columns:
    train_data[col].fillna('no item', inplace=True)
    test_data[col].fillna('no item', inplace=True)

for column in train_data.columns:
    if train_data[column].dtype == 'object':  # 문자열 데이터
        train_data[column] = train_data[column].fillna('Unknown')
    elif train_data[column].dtype in ['int64', 'float64']:  # 숫자 데이터
        train_data[column] = train_data[column].fillna(train_data[column].mean())

for column in test_data.columns:
    if test_data[column].dtype == 'object':  # 문자열 데이터
        test_data[column] = test_data[column].fillna('Unknown')
    elif test_data[column].dtype in ['int64', 'float64']:  # 숫자 데이터
        test_data[column] = test_data[column].fillna(test_data[column].mean())

# 카테고리형 및 수치형 피처 구분
categorical_features = [
    'gendr_nm', 'gen_div_nm', 'ctdo_nm', 'empmbr_entr_yn', 'lge_mbr_entr_yn', 'mbsp_app_estb_yn', 'mbsp_entr_yn', 'pref_stlm_kd',
    'thinq_entr_yn', 'new_prdc_pref_yn', 'orco_prdc_mny_buy_yn', 'orco_prdc_mny_hld_yn', 'hprop_clcn_voc_yn', 'lgebst_buy_hist_xstn_yn',
    'buy_hist_xstn_yn', 'most_infw_domn_nm', 'age_group',
    'rcnt_sub_0', 'rcnt_sub_1', 'rcnt_sub_2', 'rcnt_sub_3', 'rcnt_sub_4', 
    'rcnt_buy_0', 'rcnt_buy_1', 'rcnt_buy_2', 'rcnt_buy_3', 'rcnt_buy_4'
]
numeric_features = [
    'orco_prdc_buy_cnt', 'orco_prdc_hld_cnt', 'totl_stay_tm_sum', 
    'totl_pvw_cont_sum', 'inpc_page_knd_cont_sum'
]

# 카테고리형 변수 인코딩
for feat in categorical_features:
    lbe = LabelEncoder()
    train_data[feat] = lbe.fit_transform(train_data[feat])
    test_data[feat] = lbe.fit_transform(test_data[feat])

# 수치형 변수 스케일링
scaler = StandardScaler()
train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features])
test_data[numeric_features] = scaler.fit_transform(test_data[numeric_features])

train_dataset, val_dataset = train_test_split(train_data, test_size=0.2)

# Dataset 객체 생성
train_dataset = PandasDataset(train_data)

batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

num_features = 32
num_embedding = 32

model = FinalMLP(
   dim_input=num_features,
   num_embedding=num_embedding,
   dim_embedding=32,
   dropout=0.2,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

epochs = 100


for epoch in range(epochs):
    model.train()

    train_loss = 0.0
    for data, label in train_loader:
        data, label = data.to(device).long(), label.to(device)

        print(data.shape)

        optimizer.zero_grad()
        outputs = model(data).squeeze()
        loss = criterion(outputs, label.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)

            
            outputs = model(data).squeeze()
            loss = criterion(outputs, label.float())
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Prediction

model.eval()
y_pred = []
with torch.no_grad():
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)

        outputs = model(data).squeeze()
        y_pred.extend(outputs.cpu().numpy())
