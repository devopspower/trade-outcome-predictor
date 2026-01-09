import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

class TradeRiskDataset(Dataset):
    """
    Standard Dataset for Dual-Stream Trade Risk Data.
    """
    def __init__(self, cat_data, num_data, labels=None):
        self.cat_data = torch.tensor(cat_data, dtype=torch.long)
        self.num_data = torch.tensor(num_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.num_data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.cat_data[idx], self.num_data[idx], self.labels[idx]
        return self.cat_data[idx], self.num_data[idx]

def get_processed_trades(file_path, batch_size=32):
    """
    Cleans trade data and extracts features for loss-probability prediction.
    Target: 0 for Loss, 1 for Win (predicting the probability of the outcome).
    """
    # 1. Ground Objectively: Load Data
    df = pd.read_csv(file_path)
    
    # 2. Feature Engineering: Time-Based Analytics 
    # Convert trade_time to 'Minutes Since Market Open' (assuming 9:30 AM open)
    df['trade_time'] = pd.to_datetime(df['trade_time'], format='%H:%M')
    market_open = pd.to_datetime('09:30', format='%H:%M')
    df['mins_since_open'] = (df['trade_time'] - market_open).dt.total_seconds() / 60
    
    # Extract Hour of Day for session buckets
    df['hour_of_day'] = df['trade_time'].dt.hour
    
    # 3. Categorical Stream (Interpretation Layer) 
    cat_cols = ['day_of_week', 'market_cap_bucket', 'industry', 'hour_of_day']
    num_cols = ['mins_since_open']
    target = 'win_loss'
    
    # Initialize Encoders
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    # 4. Numerical Stream: Standardize time offsets
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # 5. Act Precisely: Save Assets for Inference
    assets = {
        'encoders': encoders,
        'scaler': scaler,
        'cat_cols': cat_cols,
        'num_cols': num_cols
    }
    joblib.dump(assets, 'trade_assets.joblib')
    
    # 6. Prepare DataLoaders
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    # Calculate Embedding Dimensions for model.py
    emb_dims = [(len(encoders[col].classes_), min(50, (len(encoders[col].classes_) + 1) // 2)) 
                for col in cat_cols]
    
    train_ds = TradeRiskDataset(train_df[cat_cols].values, train_df[num_cols].values, train_df[target].values)
    val_ds = TradeRiskDataset(val_df[cat_cols].values, val_df[num_cols].values, val_df[target].values)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, emb_dims, len(num_cols)

if __name__ == "__main__":
    # Test processing
    t_loader, v_loader, e_dims, n_feats = get_processed_trades('data/stock-market-trades.csv')
    print(f"Dataset Processed: {len(t_loader.dataset)} training samples.")
    print(f"Embedding Dimensions: {e_dims}")