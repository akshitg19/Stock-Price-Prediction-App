import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from datetime import datetime

# Set the time range (20 years of data)
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Download NVIDIA stock data
stock = "NVDA"
nvidia_data = yf.download(stock, start, end)

# Plot NVIDIA Adjusted Close Price
plt.figure(figsize=(15, 5))
nvidia_data['Adj Close'].plot()
plt.xlabel("Years")
plt.ylabel("Adjusted Close")
plt.title("NVIDIA Adjusted Close Price")
plt.show()

# Calculate 100-day and 250-day moving averages
nvidia_data['MA_for_100_days'] = nvidia_data['Adj Close'].rolling(100).mean()
nvidia_data['MA_for_250_days'] = nvidia_data['Adj Close'].rolling(250).mean()

# Normalize the adjusted close price
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(nvidia_data[['Adj Close']])

# Prepare training and test datasets (70% training, 30% testing)
look_back = 100
train_size = int(len(scaled_data) * 0.7)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - look_back:]

# Prepare the training data
x_train, y_train = [], []
for i in range(look_back, len(train_data)):
    x_train.append(train_data[i - look_back:i, 0])
    y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Prepare the test data
x_test, y_test = [], []
for i in range(look_back, len(test_data)):
    x_test.append(test_data[i - look_back:i, 0])
    y_test.append(test_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3)
model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), callbacks=[early_stop])

# Make predictions
predictions = model.predict(x_test)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(inv_y_test, inv_predictions))
mae = mean_absolute_error(inv_y_test, inv_predictions)
print(f"RMSE: {rmse}, MAE: {mae}")

# Create a DataFrame for actual and predicted values
plotting_data = pd.DataFrame({
    'Actual Price': inv_y_test.flatten(),
    'Predicted Price': inv_predictions.flatten()
})

# Fix the index to match the length of the plotting data
plotting_data.index = nvidia_data.index[-len(plotting_data):]

# Plot actual vs predicted prices
plt.figure(figsize=(15, 6))
plt.plot(plotting_data['Actual Price'], label='Actual Price')
plt.plot(plotting_data['Predicted Price'], label='Predicted Price')
plt.title('NVIDIA Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Plot full data including training and prediction
full_data = pd.concat([nvidia_data['Adj Close'][:train_size + look_back], plotting_data], axis=0)
plt.figure(figsize=(15, 6))
plt.plot(full_data['Adj Close'], label='Training Data')
plt.plot(full_data['Actual Price'], label='Actual Price')
plt.plot(full_data['Predicted Price'], label='Predicted Price')
plt.title('NVIDIA Stock Price Prediction (Full Data)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Save the model
model.save("Latest_stock_price_model.keras")
