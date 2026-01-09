import streamlit as st
import torch
import joblib
import pandas as pd
import datetime
from model import TradeRiskNet
from inference import safe_transform

# --- Page Configuration ---
st.set_page_config(
    page_title="Trade Risk Guard",
    page_icon="🛡️",
    layout="centered"
)

# --- Asset Loading (Optimized with Caching) ---
@st.cache_resource
def load_model_assets():
    try:
        # Load preprocessing metadata
        assets = joblib.load('trade_assets.joblib')
        encoders = assets['encoders']
        scaler = assets['scaler']
        cat_cols = assets['cat_cols']
        num_cols = assets['num_cols']
        
        # Determine architecture dimensions from saved encoders
        emb_dims = [(len(encoders[col].classes_), min(50, (len(encoders[col].classes_) + 1) // 2)) 
                    for col in cat_cols]
        
        # Initialize and load model weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TradeRiskNet(emb_dims, len(num_cols)).to(device)
        model.load_state_dict(torch.load('trade_model.pth', map_location=device))
        model.eval()
        
        return model, assets, device
    except FileNotFoundError:
        st.error("Required files (trade_model.pth or trade_assets.joblib) not found. Please run main.py first.")
        return None, None, None

def main():
    st.title("🛡️ Trade Outcome Risk Predictor")
    st.markdown("""
    **Goal:** Identify high-risk market conditions to reduce losses. 
    *This AI analyzes historical 'Worst Case' scenarios to support your trade decisions.*
    """)

    model, assets, device = load_model_assets()
    if not model: return

    # --- Sidebar Input Section ---
    st.sidebar.header("📝 Trade Conditions")
    
    with st.sidebar:
        day = st.selectbox("Day of Week", list(assets['encoders']['day_of_week'].classes_))
        mkt_cap = st.selectbox("Market Cap Bucket", list(assets['encoders']['market_cap_bucket'].classes_))
        industry = st.selectbox("Industry Type", list(assets['encoders']['industry'].classes_))
        
        st.divider()
        st.subheader("⏰ Timing Analysis")
        # Default to 9:30 AM market open
        trade_time = st.time_input("Planned Trade Execution Time", value=datetime.time(10, 0))
        
        # Calculate minutes since market open (9:30 AM)
        market_open_total_mins = 9 * 60 + 30
        input_total_mins = trade_time.hour * 60 + trade_time.minute
        mins_since_open = float(input_total_mins - market_open_total_mins)
        
        hour_of_day = trade_time.hour

    # --- Inference Logic ---
    if st.sidebar.button("Analyze Risk Profile", type="primary"):
        # 1. Prepare Inputs
        input_dict = {
            'day_of_week': day,
            'market_cap_bucket': mkt_cap,
            'industry': industry,
            'hour_of_day': hour_of_day,
            'mins_since_open': mins_since_open
        }
        
        # Transform categorical
        cat_values = [safe_transform(assets['encoders'][col], input_dict[col]) for col in assets['cat_cols']]
        
        # Transform numerical
        num_scaled = assets['scaler'].transform([[mins_since_open]])
        
        # 2. Compute Probability
        cat_tensor = torch.tensor([cat_values], dtype=torch.long).to(device)
        num_tensor = torch.tensor(num_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(cat_tensor, num_tensor)
            win_prob = torch.sigmoid(logits).item()
            loss_prob = 1 - win_prob

        # --- Display Results ---
        st.divider()
        
        # Visual Meter
        st.subheader("Risk Interpretation")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Loss Probability", f"{loss_prob:.2%}")
        
        with col2:
            if loss_prob > 0.65:
                st.error("### 🚨 CRITICAL RISK CONDITION")
                st.write(f"Historical data shows high structural loss rates for **{industry}** stocks on **{day}s** at this time.")
            elif loss_prob > 0.45:
                st.warning("### ⚠️ ELEVATED RISK")
                st.write("Conditions are volatile. Recommend reducing position size or tightening stop-losses.")
            else:
                st.success("### ✅ FAVORABLE WINDOW")
                st.write("Historical risk for this combination is low. Standard execution protocols apply.")

        # --- Contextual Insight ---
        with st.expander("See Feature Analysis"):
            st.write(f"- **Execution Window:** {mins_since_open:.0f} minutes after market open.")
            st.write(f"- **Market Session:** Hour {hour_of_day} on {day}.")
            st.write(f"- **Sector Exposure:** {industry} ({mkt_cap} cap).")

    else:
        st.write("---")
        st.info("Fill in the trade details in the sidebar and click **Analyze Risk Profile** to start.")

if __name__ == "__main__":
    main()