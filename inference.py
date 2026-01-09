import torch
import joblib
import numpy as np
from model import TradeRiskNet

def safe_transform(encoder, value):
    """
    Safely retrieves the integer index for categorical trade data.
    Ensures that unknown industries or days don't break the prediction flow.
    """
    if encoder.classes_.size == 0:
        return 0
    
    val_str = str(value)
    # Fallback logic: If value is unknown, use 'Unknown' or the first class
    if val_str not in encoder.classes_:
        val_str = 'Unknown' if 'Unknown' in encoder.classes_ else str(encoder.classes_[0])
    
    indices = np.where(encoder.classes_ == val_str)[0]
    return int(indices[0])

def get_risk_probability(trade_data, model_path='trade_model.pth', assets_path='trade_assets.joblib'):
    """
    Scores a potential trade to identify risk conditions.
    Returns: Probability of a Win (percentage).
    """
    # 1. Load Processing Assets
    assets = joblib.load(assets_path)
    encoders = assets['encoders']
    scaler = assets['scaler']
    cat_cols = assets['cat_cols']
    num_cols = assets['num_cols']
    
    # 2. Reconstruct Architecture Dimensions
    emb_dims = [(len(encoders[col].classes_), min(50, (len(encoders[col].classes_) + 1) // 2)) 
                for col in cat_cols]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradeRiskNet(emb_dims, len(num_cols)).to(device)
    
    # Load Best Model Weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Categorical Processing (Interpretation Layer)
    cat_values = [safe_transform(encoders[col], trade_data.get(col, 'Unknown')) for col in cat_cols]
    
    # 4. Numerical Processing (Time Analytics)
    # Ensure numerical features are list of lists for scaler
    num_values = [float(trade_data.get(col, 0)) for col in num_cols]
    num_scaled = scaler.transform([num_values])

    # 5. Predict Risk
    cat_tensor = torch.tensor([cat_values], dtype=torch.long).to(device)
    num_tensor = torch.tensor(num_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(cat_tensor, num_tensor)
        win_probability = torch.sigmoid(logits).item()

    return win_probability

if __name__ == "__main__":
    # Test Scenario: Evaluating a potential trade during a high-risk window
    # Example: SME in Technology on a Monday afternoon
    test_trade = {
        'day_of_week': 'Monday',
        'market_cap_bucket': 'SME',
        'industry': 'Technology',
        'hour_of_day': 14,
        'mins_since_open': 285.0  # 4 hours and 45 mins after open
    }

    try:
        win_prob = get_risk_probability(test_trade)
        loss_prob = 1 - win_prob
        
        print(f"\n--- Trade Risk Analysis ---")
        print(f"Win Probability: {win_prob:.2%}")
        print(f"Loss Probability: {loss_prob:.2%}")
        
        if loss_prob > 0.60:
            print("Action: ⚠️ AVOID TRADE (High Risk Condition)")
        elif loss_prob > 0.40:
            print("Action: ⚖️ CAUTION (Elevated Risk)")
        else:
            print("Action: ✅ PROCEED (Historical Low Risk)")
            
    except Exception as e:
        print(f"Inference Error: {e}")