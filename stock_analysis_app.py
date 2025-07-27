# stock_analysis_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import LSTM, Dense, Dropout # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt

# --- Helper Functions ---

# Fetch stock data
@st.cache_data
def get_stock_data(ticker, period='1y', interval='1d'):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        if data.empty or 'Close' not in data.columns:
            return None
        return data
    except:
        return None

# Calculate daily percentage change
def calculate_daily_change(data):
    data['Daily Change'] = data['Close'].pct_change() * 100
    return data

# Plot daily profit/loss chart
def plot_daily_change(data, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Daily Change'], mode='lines', name='Daily % Change'))
    fig.update_layout(title=f'Daily % Change - {symbol}', xaxis_title='Date', yaxis_title='Change (%)')
    return fig

# Plot candlestick chart
def plot_candlestick(data, symbol):
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])
    fig.update_layout(title=f'Candlestick Chart - {symbol}', xaxis_title='Date', yaxis_title='Price')
    return fig

# Prepare data for LSTM
def prepare_lstm_data(data, look_back=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[['Close']])

    x, y = [], []
    for i in range(look_back, len(scaled)):
        x.append(scaled[i-look_back:i, 0])
        y.append(scaled[i, 0])

    x = np.array(x).reshape(-1, look_back, 1)
    y = np.array(y)
    return x, y, scaler

# Build LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict future prices
def forecast_future(model, last_sequence, days, scaler):
    predictions = []
    seq = last_sequence.copy()
    for _ in range(days):
        pred = model.predict(seq.reshape(1, -1, 1), verbose=0)[0][0]
        predictions.append(pred)
        seq = np.append(seq[1:], pred)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Plot prediction
def plot_forecast(forecast, last_date):
    future_dates = pd.date_range(start=last_date, periods=len(forecast) + 1, freq='D')[1:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=forecast.flatten(), mode='lines+markers', name='Forecast'))
    fig.update_layout(title='📈 Future Price Forecast', xaxis_title='Date', yaxis_title='Price')
    return fig

# --- Streamlit UI ---

st.set_page_config(page_title="📊 Full Stock Analysis", layout="wide")
st.title("📊 Stock Analysis & LSTM Forecasting App")

# Inputs
stock_symbol = st.text_input("Enter stock symbol (e.g. AAPL, TSLA, GOOGL):", value="AAPL").upper()
investment_amount = st.number_input("Investment Amount ($):", min_value=10.0, value=1000.0)
forecast_days = st.slider("Forecast Days into the Future", min_value=5, max_value=30, value=10)
period = st.selectbox("Select Historical Data Period", options=['1y', '2y', '5y'], index=0)

if stock_symbol:
    with st.spinner(f"Fetching data for {stock_symbol}..."):
        data = get_stock_data(stock_symbol, period=period)
    
    if data is None:
        st.error("Failed to load stock data. Please check the symbol.")
    else:
        st.success(f"Data loaded for {stock_symbol} ✅")
        data = calculate_daily_change(data)

        # Show raw data
        st.subheader("📄 Latest Stock Data")
        st.dataframe(data.tail())

        # Charts
        st.subheader("📈 Daily % Change")
        st.plotly_chart(plot_daily_change(data, stock_symbol), use_container_width=True)

        st.subheader("📉 Candlestick Chart")
        st.plotly_chart(plot_candlestick(data, stock_symbol), use_container_width=True)

        # Investment Simulation
        if 'Daily Change' in data.columns:
            cumulative = (1 + data['Daily Change'].fillna(0) / 100).cumprod()
            final_val = investment_amount * cumulative.iloc[-1]

            st.subheader("💵 Investment Performance")
            st.write(f"Initial Investment: ${investment_amount:,.2f}")
            st.write(f"Value After {period}: ${final_val:,.2f}")

            if final_val > investment_amount:
                st.success("Your investment gained value ✅")
            elif final_val < investment_amount:
                st.error("Your investment lost value ❌")
            else:
                st.info("No change in investment.")

        # LSTM Forecasting
        st.subheader("🔮 Price Forecast using LSTM")
        look_back = 60
        x_data, y_data, scaler = prepare_lstm_data(data, look_back=look_back)
        model = build_model((x_data.shape[1], 1))

        with st.spinner("Training model..."):
            model.fit(x_data, y_data, epochs=20, batch_size=32, verbose=0)

        forecast = forecast_future(model, x_data[-1].flatten(), forecast_days, scaler)
        st.plotly_chart(plot_forecast(forecast, data.index[-1]), use_container_width=True)

        st.subheader("📋 Forecasted Prices")
        for i, price in enumerate(forecast):
            st.write(f"Day {i+1}: ${price[0]:.2f}")
