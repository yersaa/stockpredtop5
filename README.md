# Halyk Bank Stock Price Prediction


This project implements machine learning and deep learning models to predict stock prices for Halyk Bank using historical financial data. The system processes historical stock data, implements various neural network architectures, and provides both short-term and long-term price forecasts.

## Features

- **Data Processing**: Handles date sorting, volume data cleaning, and missing values
- **Multiple Models**: Implements MLP, RNN, and LSTM architectures
- **Time Series Analysis**: Utilizes moving averages and rolling windows for trend analysis
- **Future Prediction**: Forecasts stock prices for next day, next 7 days, and next 6 months
- **Visualization**: Includes price history charts, moving average comparisons, and prediction plots

## Key Results

| Model | Training RMSE | Testing RMSE |
|-------|---------------|--------------|
| LSTM | 0.30 | 0.40 |
| RNN | 0.33 | 0.46 |
| MLP Model 1 | 0.57 | 2.24 |
| MLP Model 2 | 4.40 | 6.37 |

**Next 7 Days Predictions**:
- 2025-06-14: 23.18
- 2025-06-15: 23.27
- 2025-06-16: 23.23
- 2025-06-17: 23.18
- 2025-06-18: 23.13
- 2025-06-19: 23.08
- 2025-06-20: 23.04

## Requirements

- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - torch
  - scikit-learn
  - matplotlib
  - seaborn
  - d21 (teaching library)

## Installation

```bash
git clone https://github.com/yourusername/halyk-stock-prediction.git
cd halyk-stock-prediction
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**:
   - Place your stock data in `data/halyk_stock_price.csv`
   - Run the preprocessing script to clean and prepare data

2. **Training Models**:
```python
# From notebook:
# Model initialization
lstm_model = LSTM(input_dim=1, hidden_dim=10, num_layers=1, output_dim=1)

# Model training
for epoch in range(num_epochs):
    y_train_pred = lstm_model(X_train)
    loss = criterion(y_train_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

3. **Making Predictions**:
```python
# Predict next day
predicted_price = predict_next_day(lstm_model, last_sequence, scaler)

# Predict next 7 days
next_7_days = predict_next_n_days(lstm_model, initial_sequence, scaler, 7)
```

## Project Structure

```
halyk-stock-prediction/
├── data/                   # Dataset directory
│   └── halyk_stock_price.csv
├── notebooks/              # Jupyter notebooks
│   └── Stock_Prediction.ipynb
├── src/                    # Source code
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── prediction.py
├── models/                 # Saved models (optional)
├── requirements.txt        # Dependencies
└── README.md
```

## Model Architectures

### LSTM (Long Short-Term Memory)
```python
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
```

### MLP (Multi-Layer Perceptron)
```python
class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(num_inputs, num_hiddens)
        self.activation = nn.ReLU()
        self.out = nn.Linear(num_hiddens, num_outputs)
    
    def forward(self, X):
        X = self.hidden(X)
        X = self.activation(X)
        X = self.out(X)
        return X
```



## Key Insights

1. **Best Performing Model**: LSTM achieved the lowest test RMSE (0.40)
2. **Prediction Horizon**: 
   - Short-term predictions (7 days) are more accurate than long-term (6 months)
   - Model predicts a gradual decline over the next 6 months
3. **Data Characteristics**:
   - Date range: 2012-08-30 to 2025-06-13
   - Average closing price: 10.94
   - Highest closing price: 26.40

## Recommendations

1. Use LSTM model for short-term trading decisions
2. Monitor key technical indicators along with model predictions
3. Re-train models monthly with new market data
4. Combine predictions with fundamental analysis for investment decisions
5. Implement risk management strategies considering model uncertainty

