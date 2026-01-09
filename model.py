import torch
import torch.nn as nn

class TradeRiskNet(nn.Module):
    """
    Deep Neural Network for Trade Outcome Classification.
    Uses Entity Embeddings for categorical risk factors and 
    dense layers for numerical features.
    """
    def __init__(self, emb_dims, n_num):
        super(TradeRiskNet, self).__init__()
        
        # 1. Analyze Logically: Categorical Embedding Stream 
        # Maps Day of Week, Market Cap, and Industry into latent space
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) 
            for num_classes, emb_dim in emb_dims
        ])
        
        # Calculate total dimensions after embedding concatenation
        n_emb = sum(emb_dim for _, emb_dim in emb_dims)
        
        # 2. Analyze Logically: Deep Learning Architecture 
        # Multi-layer Perceptron (MLP) for non-linear pattern recognition
        self.mlp = nn.Sequential(
            nn.Linear(n_emb + n_num, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),  # Dropout to prevent overfitting to market noise
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, 1)  # Single logit output for binary classification
        )

    def forward(self, x_cat, x_num):
        # Embed all categorical inputs and concatenate
        embeddings = [
            emb(x_cat[:, i]) 
            for i, emb in enumerate(self.embeddings)
        ]
        x_emb = torch.cat(embeddings, dim=1)
        
        # Combine with numerical features (Minutes since open, etc.)
        x = torch.cat([x_emb, x_num], dim=1)
        
        # Pass through the MLP head
        logits = self.mlp(x)
        return logits.squeeze()