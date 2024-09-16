import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.title("Stock Price Predictor App")
st.write("Enter a stock symbol to view predictions and historical data.")

# Stock Input
stock = st.text_input("Enter the Stock ID", "GOOG")

# Download stock data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
google_data = yf.download(stock, start, end)

# Load the pre-trained model
model = load_model("Latest_stock_price_model.keras")

# Display stock data
st.subheader("Stock Data")
st.write(google_data)

# Moving averages plot function
def plot_graph(values, label, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(values, label=label)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

# Calculate and plot Moving Averages
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()

st.subheader('Moving Averages')
plot_graph(google_data['MA_for_250_days'], 'MA for 250 days', '250-Day Moving Average')
plot_graph(google_data['MA_for_200_days'], 'MA for 200 days', '200-Day Moving Average')
plot_graph(google_data['MA_for_100_days'], 'MA for 100 days', '100-Day Moving Average')

# Data preparation for predictions
splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

# Scale data for predictions
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

# Prepare input sequences for model
x_data = []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])

x_data = np.array(x_data)

# Predict with the model
predictions = model.predict(x_data)

# Inverse transform to get original values
inv_predictions = scaler.inverse_transform(predictions)

# Create a DataFrame for comparison
plotting_data = pd.DataFrame({
    'Actual': google_data.Close[splitting_len + 100:],
    'Predicted': inv_predictions.flatten()
})

# Display comparison
st.subheader("Original vs Predicted Stock Prices")
st.write(plotting_data)

# Plot actual vs predicted
st.subheader('Actual vs Predicted Prices')
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(plotting_data.index, plotting_data['Actual'], label='Actual Prices', color='blue')
ax.plot(plotting_data.index, plotting_data['Predicted'], label='Predicted Prices', color='orange')
ax.set_title('Stock Price Prediction')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)
