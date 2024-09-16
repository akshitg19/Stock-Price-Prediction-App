Stock Price Predictor App
A Python-based web application that predicts stock prices using a pre-trained LSTM model. The app fetches real-time stock data, analyzes trends, and visualizes both historical and predicted stock prices.

Features
Real-time stock data: Fetches 20 years of historical data using the yFinance API.
Moving Averages: Calculates and plots 100, 200, and 250-day moving averages.
Predictive Model: Uses an LSTM model to predict future stock prices.
Data Visualization: Displays actual vs. predicted prices using Matplotlib.
Tech Stack
Frontend: Streamlit
Backend: Python
Machine Learning: LSTM (Long Short-Term Memory)
APIs: yFinance for stock data
Libraries: Pandas, NumPy, Matplotlib, Scikit-learn
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/stock-price-predictor.git
cd stock-price-predictor
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the application:

bash
Copy code
streamlit run app.py
Usage
Enter a valid stock symbol (e.g., AAPL, TSLA) in the app.
View historical stock data, moving averages, and predictions.
Compare actual vs. predicted stock prices.
