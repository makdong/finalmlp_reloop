import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from torch.utils.data import DataLoader, Dataset
import torch

from utils import CustomDataset

class DataPreprocessor:
    def __init__(self, train_path, test_path, batch_size=4096, type="train"):
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size

        self.categorical_features = [
            'gendr_nm', 'gen_div_nm', 'ctdo_nm', 'empmbr_entr_yn', 'lge_mbr_entr_yn', 'mbsp_app_estb_yn', 'mbsp_entr_yn', 'pref_stlm_kd',
            'thinq_entr_yn', 'new_prdc_pref_yn', 'orco_prdc_mny_buy_yn', 'orco_prdc_mny_hld_yn', 'hprop_clcn_voc_yn', 'lgebst_buy_hist_xstn_yn',
            'buy_hist_xstn_yn', 'most_infw_domn_nm', 'age_group',
            'rcnt_sub_0', 'rcnt_sub_1', 'rcnt_sub_2', 'rcnt_sub_3', 'rcnt_sub_4', 
            'rcnt_buy_0', 'rcnt_buy_1', 'rcnt_buy_2', 'rcnt_buy_3', 'rcnt_buy_4'
        ]
        self.numeric_features = [
            'orco_prdc_buy_cnt', 'orco_prdc_hld_cnt', 'totl_stay_tm_sum', 
            'totl_pvw_cont_sum', 'inpc_page_knd_cont_sum'
        ]
        self.rcnt_columns = [
            'rcnt_sub_0', 'rcnt_sub_1', 'rcnt_sub_2', 'rcnt_sub_3', 'rcnt_sub_4',
            'rcnt_buy_0', 'rcnt_buy_1', 'rcnt_buy_2', 'rcnt_buy_3', 'rcnt_buy_4'
        ]
        self.train_data = None
        self.test_data = None
        self.type = type

    def load_data(self):
        """Load train and test datasets."""
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)

    def clean_data(self):
        """Remove unnecessary columns and handle missing values."""
        drop_columns = ['train/test', 'cust_id', 'base_dt']
        self.train_data = self.train_data.drop(columns=drop_columns)
        self.test_data = self.test_data.drop(columns=drop_columns)

        # Fill missing values for specific columns
        for col in self.rcnt_columns:
            self.train_data[col].fillna('no item', inplace=True)
            self.test_data[col].fillna('no item', inplace=True)

        # Fill remaining missing values
        for dataset in [self.train_data, self.test_data]:
            for column in dataset.columns:
                if dataset[column].dtype == 'object':
                    dataset[column] = dataset[column].fillna('Unknown')
                elif dataset[column].dtype in ['int64', 'float64']:
                    dataset[column] = dataset[column].fillna(dataset[column].mean())

    def encode_and_scale(self, pc_tv_indices, non_pc_tv_indices):
        """Encode categorical variables and scale numeric variables."""
    
        obe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=99, encoded_missing_value=999)
        self.train_data[self.categorical_features] = obe.fit_transform(self.train_data[self.categorical_features])
        self.test_data[self.categorical_features] = obe.transform(self.test_data[self.categorical_features])

        scaler = StandardScaler()
        self.train_data[self.numeric_features] = scaler.fit_transform(self.train_data[self.numeric_features])
        self.test_data[self.numeric_features] = scaler.transform(self.test_data[self.numeric_features])

        data_with_pc_tv = self.train_data.loc[pc_tv_indices].reset_index(drop=True)
        data_without_pc_tv = self.train_data.loc[non_pc_tv_indices].reset_index(drop=True)

        return data_with_pc_tv, data_without_pc_tv

    def mask_data(self):
        """Split data into subsets based on specific conditions."""
        mask = self.train_data[self.rcnt_columns].apply(lambda row: row.isin(['PC', 'TV']).any(), axis=1)
        pc_tv_indices = self.train_data[mask].index
        non_pc_tv_indices = self.train_data[~mask].index

        return pc_tv_indices, non_pc_tv_indices

    def create_dataloaders(self, data_with_pc_tv, data_without_pc_tv):
        """Create PyTorch DataLoader objects."""
        train_dataset = CustomDataset(data_without_pc_tv)
        val_dataset = CustomDataset(data_with_pc_tv)
        test_dataset = CustomDataset(self.test_data)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def process(self):
        """Run the full preprocessing pipeline."""
        self.load_data()
        self.clean_data()

        pc_tv_indices, non_pc_tv_indices = self.mask_data()
        data_with_pc_tv, data_without_pc_tv = self.encode_and_scale(pc_tv_indices, non_pc_tv_indices)

        ### this part should be deleted if adapted in ALO system. Just send the dataloader via argument.
        data_without_pc_tv.to_csv('localdata/data_without_pctv.csv', index=False)
        data_with_pc_tv.to_csv('localdata/data_with_pctv.csv', index=False)
        self.test_data.to_csv('localdata/test.csv', index=False)

        return self.create_dataloaders(data_with_pc_tv, data_without_pc_tv)
    

preprocessor = DataPreprocessor('train_B2C.csv', 'test_B2C.csv', batch_size=4096)
train_loader, val_loader, test_loader = preprocessor.process()
