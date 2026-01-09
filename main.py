import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score
from trade_processor import get_processed_trades
from model import TradeRiskNet
import numpy as np

def train_risk_model():
    # 1. Ground Objectively: Load processed data and model dimensions
    # batch_size is set to 32 as per the processor design
    train_loader, val_loader, emb_dims, n_num = get_processed_trades('data/stock-market-trades.csv', batch_size=32)
    
    # 2. Analyze Logically: Initialize Architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradeRiskNet(emb_dims, n_num).to(device)
    
    # Loss focused on binary classification: 0 (Loss) vs 1 (Win)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    epochs = 20
    best_auc = 0.0
    
    print(f"--- Trade Risk Training Pipeline ---")
    print(f"Target Device: {device}")
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for x_cat, x_num, y in train_loader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x_cat, x_num)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        # 3. Validate Rigorously: Evaluation Phase
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x_cat, x_num, y in val_loader:
                x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
                logits = model(x_cat, x_num)
                probs = torch.sigmoid(logits) # Convert logits to 0-1 probability
                
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        # Calculate Metrics
        auc = roc_auc_score(all_targets, all_preds)
        binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
        acc = accuracy_score(all_targets, binary_preds)
        
        avg_train_loss = np.mean(train_losses)
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Loss: {avg_train_loss:.4f} | ROC-AUC: {auc:.4f} | Acc: {acc:.4f}")
        
        # Save the best model based on ROC-AUC (discriminative power)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'trade_model.pth')
            
    print("--- Training Complete ---")
    print(f"Best Validation ROC-AUC Achieved: {best_auc:.4f}")
    print("Model serialized to: trade_model.pth")

if __name__ == "__main__":
    train_risk_model()