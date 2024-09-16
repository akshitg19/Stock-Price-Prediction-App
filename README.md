# Stock Price Predictor App

A Python-based web application that predicts stock prices using a pre-trained LSTM model. The app fetches real-time stock data, analyzes trends, and visualizes both historical and predicted stock prices.

## Features
- **Real-time stock data**: Fetches 20 years of historical data using the yFinance API.
- **Moving Averages**: Calculates and plots 100, 200, and 250-day moving averages.
- **Predictive Model**: Uses an LSTM model to predict future stock prices.
- **Data Visualization**: Displays actual vs. predicted prices using Matplotlib.

## Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: LSTM (Long Short-Term Memory)
- **APIs**: yFinance for stock data
- **Libraries**: Pandas, NumPy, Matplotlib, Scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-price-predictor.git
   cd stock-price-predictor
