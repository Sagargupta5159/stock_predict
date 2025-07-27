import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- Fetch stock data with fallback ---
def get_stock_data(ticker, period='1y', interval='1d'):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval, actions=False)
        
        if data.empty:
            st.warning(f"No data returned for {ticker} with 1y/1d. Trying 6mo/1d as fallback...")
            data = stock.history(period='6mo', interval='1d', actions=False)
        
        if data.empty:
            st.warning(f"Fallback also failed. Trying 1y/1wk for {ticker}...")
            data = stock.history(period='1y', interval='1wk', actions=False)

        return data if not data.empty else None

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# --- Candlestick chart ---
def plot_candlestick(data, stock_symbol):
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])
    fig.update_layout(
        title=f'Candlestick Chart - {stock_symbol}',
        xaxis_title='Date',
        yaxis_title='Stock Price'
    )
    return fig

# --- Calculate daily change ---
def calculate_daily_profit_loss(data):
    data['Daily Change'] = data['Close'].pct_change() * 100
    return data

# --- Line chart for daily profit/loss ---
def plot_profit_loss(data, stock_symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Daily Change'],
        mode='lines',
        name='Daily % Change'
    ))
    fig.update_layout(
        title=f'Daily Profit/Loss - {stock_symbol}',
        xaxis_title='Date',
        yaxis_title='Daily % Change',
        yaxis_tickformat='.2f'
    )
    return fig

# --- Streamlit UI ---
st.title('📊 Stock Analysis Project')

stocks = [
    'AAPL', 'GOOGL', 'MSFT', 'COIN', 'AMZN', 'MRNA', 'NVAX', 'INO', 'ARM',
    'BNTX', 'VXRT', 'NCLH', 'AZN', 'TCS', 'LBPH', 'AMD', 'HIMS', 'FIZZ',
    'CRSP', 'LPL', 'PRTA', 'USCA', 'USAC', 'YY', 'LYFT', 'LI', 'NIO', 'JOBY', 'SQ'
]

selected_stocks = st.multiselect('Select stock symbols:', stocks)
investment_amount = st.number_input('Enter your investment amount:', min_value=0.01, value=1000.0, step=0.01)

stock_data_dict = {}

# --- Fetch and process data for selected stocks ---
for stock_symbol in selected_stocks:
    with st.spinner(f'Fetching stock data for {stock_symbol}...'):
        data = get_stock_data(stock_symbol)
    
    if data is None or data.empty:
        st.error(f"No data available for {stock_symbol}. Please check the symbol or try again later.")
    else:
        data = calculate_daily_profit_loss(data)
        stock_data_dict[stock_symbol] = data
        st.success(f'Data fetched for {stock_symbol}.')

# --- Display results ---
for stock_symbol, stock_data in stock_data_dict.items():
    st.subheader(f'📈 Stock Data - {stock_symbol}')
    st.write(stock_data.tail())

    st.subheader(f'📉 Daily Profit/Loss - {stock_symbol}')
    st.plotly_chart(plot_profit_loss(stock_data, stock_symbol))

    st.subheader(f'🕯️ Candlestick Chart - {stock_symbol}')
    st.plotly_chart(plot_candlestick(stock_data, stock_symbol))

    # Investment simulation
    if len(stock_data) > 1:
        returns = stock_data['Daily Change'].fillna(0) / 100
        cumulative_returns = (1 + returns).cumprod()
        final_value = investment_amount * cumulative_returns.iloc[-1]

        st.subheader(f'💰 Investment Summary - {stock_symbol}')
        st.write(f'Initial Investment: **${investment_amount:.2f}**')
        st.write(f'Final Value: **${final_value:.2f}**')

        total_return = ((final_value - investment_amount) / investment_amount) * 100

        if total_return > 0:
            st.success(f'✅ You gained a profit of **{total_return:.2f}%**')
        elif total_return < 0:
            st.error(f'🔻 You incurred a loss of **{abs(total_return):.2f}%**')
        else:
            st.info('⚖️ No profit, no loss. The stock price remained unchanged.')
    else:
        st.warning(f"Not enough data to simulate investment performance for {stock_symbol}.")
