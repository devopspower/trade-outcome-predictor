# AI-Driven Trade Outcome Predictor (Risk Guard)

An end-to-end deep learning solution designed for **loss reduction and decision support** in stock market trading. Unlike traditional price predictors, this system implements a **Dual-Stream Neural Network** in PyTorch to identify high-risk historical conditions, allowing traders to avoid "Worst-Case" scenarios based on time, day, and industry.

## 📈 Project Performance

Based on the training execution using the `stock-market-trades.csv` dataset:

- **Best Validation ROC-AUC:** **0.9242**
- **Peak Accuracy:** **85.00%** (Reached by Epoch 2)
- **Loss Convergence:** Reduced from **0.6963** to **0.3667** within 9 epochs.
- **Optimization:** The model demonstrates exceptional discriminative power, effectively separating high-probability "Loss" conditions from "Win" conditions.

## 🛠️ Step-by-Step Methodology

### 1. Step 1 (Objective)

The goal was framed as a **Binary Classification** problem focusing on risk avoidance. Instead of chasing returns, the objective was to predict the **probability of a loss** (Outcome: 0) to support decision-making and minimize the number of losing trades.

### 2. Step 2 (Data Engineering)

- **Time Analytics:** Converted raw timestamps into "Minutes since Market Open" to capture volatility windows relative to the 9:30 AM open.
- **Feature Selection:** Utilized `day_of_week`, `market_cap_bucket`, `industry`, and `hour_of_day` to build an interpretation layer of structural risk.
- **Normalization:** Applied `StandardScaler` to numerical time features and `LabelEncoder` for categorical risk factors.

### 3. Step 3 (Architecture)

- **Entity Embeddings:** Created high-dimensional vector representations for categorical data (Industry, Day, Cap Size) to find non-linear risk correlations.
- **Deep MLP Head:** A multi-layer perceptron with **Batch Normalization** and **Dropout (0.3)** to prevent overfitting to historical market noise.
- **Activation:** Used **BCEWithLogitsLoss** for numerical stability during training.

### 4. Step 4 (Validation)

- **ROC-AUC Metric:** Prioritized ROC-AUC to ensure the model accurately ranks the probability of risk across all potential thresholds.
- **Safe-Lookup Logic:** Implemented an inference fallback mechanism to handle unseen or "Unknown" categories during live testing.

### 5. Step 5 (Interactive Deployment)

Created a **Streamlit Dashboard** that provides:

- **Risk Scoring:** Real-time probability of loss for potential trade entries.
- **Decision Support:** Automated "Avoid/Caution/Proceed" alerts based on historical loss magnitudes.

## 📂 File Structure

- `trade_processor.py`: Time-offset calculation and data engineering pipeline.
- `model.py`: PyTorch architecture with Categorical Embedding streams.
- `main.py`: Training orchestrator focused on ROC-AUC optimization.
- `inference.py`: Standalone scoring logic with safe categorical transformation.
- `app.py`: Interactive Streamlit dashboard for risk visualization.

## 🚀 Getting Started

1. **Install Dependencies:**

```bash
pip install -r requirements.txt

```

2. **Run Training:**

```bash
python main.py

```

3. **Launch Risk Dashboard:**

```bash
streamlit run app.py

```

## 💡 Strategic Insights

- **Interpretation Layer:** The model identified that specific industries combined with "late-session" volatility significantly increase loss probability.
- **Loss Mitigation:** By filtering out trades with a >60% predicted loss probability, the system serves as a "Guard" for capital preservation.
