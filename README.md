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
   git clone https://github.com/akshitg19/Stock-Price-Prediction-App
   cd stock-price-predictor

2. Install the required dependencies for the stock predictor model:
   ```bash
   pip install scikit-learn
   python stock_predictor.py
   ```

3. Install Streamlit for the web app interface:
   ```bash
   pip install streamlit
   ```

4. Run the web app:
   ```bash
   streamlit run web_app.py
   ```


   ![image](https://github.com/user-attachments/assets/577434fd-d3e1-49e2-80a8-8e354b710fab)
   ![image](https://github.com/user-attachments/assets/e01b44c4-1b9d-44e6-ac51-bde7e5894ae7)
   ![image](https://github.com/user-attachments/assets/40f9ff49-2eed-4f9e-8073-c0ced6c4ae86)



