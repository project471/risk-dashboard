import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Risk Dashboard", layout="wide")
st.title("📊 Risk Management Dashboard (VaR, CVaR & Prediction)")

# -----------------------------
# User Inputs
# -----------------------------
ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95)

# -----------------------------
# Fetch Data (Robust)
# -----------------------------
@st.cache_data(ttl=300)
def fetch_data(symbol):
    df = yf.download(symbol, period="6mo", interval="1d", progress=False)

    if df.empty:
        return None

    # Fix multi-index columns (important)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        return None

    df = df[required_cols].copy()

    # Convert to numeric safely
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop NaN rows
    df = df.dropna()

    df = df.rename(columns={'Close': 'close'})
    df = df.reset_index()

    return df

data = fetch_data(ticker)

if data is None or data.empty:
    st.error("Invalid ticker or no data available.")
    st.stop()

# -----------------------------
# Returns & Risk Metrics
# -----------------------------
data['Returns'] = data['close'].pct_change()
returns = data['Returns'].dropna()

if len(returns) == 0:
    st.error("Not enough data to calculate returns.")
    st.stop()

var_hist = np.percentile(returns, (1 - confidence_level) * 100)
var_param = norm.ppf(1 - confidence_level, returns.mean(), returns.std())
cvar = returns[returns <= var_hist].mean()

data['Volatility'] = data['Returns'].rolling(30).std()

# -----------------------------
# Metrics
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Historical VaR", round(var_hist, 6))
col2.metric("Parametric VaR", round(var_param, 6))
col3.metric("CVaR", round(cvar, 6))

# -----------------------------
# 1. Candlestick Price Chart
# -----------------------------
fig1 = go.Figure()

fig1.add_trace(go.Candlestick(
    x=data['Date'],
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['close'],
    increasing_line_color='green',
    decreasing_line_color='red'
))

fig1.update_layout(
    title="Stock Price (Candlestick)",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# 2. Returns Distribution + VaR
# -----------------------------
fig2 = go.Figure()

fig2.add_trace(go.Histogram(
    x=returns,
    nbinsx=50,
    name="Returns",
    marker_color='blue'
))

fig2.add_vline(
    x=var_hist,
    line_color="red",
    annotation_text="VaR",
    annotation_position="top left"
)

fig2.update_layout(
    title="Returns Distribution with VaR",
    xaxis_title="Daily Returns",
    yaxis_title="Frequency"
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 3. Rolling Volatility
# -----------------------------
fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=data['Date'],
    y=data['Volatility'],
    fill='tozeroy',
    mode='lines',
    line=dict(color='purple'),
    name="Volatility"
))

fig3.update_layout(
    title="30-Day Rolling Volatility",
    xaxis_title="Date",
    yaxis_title="Volatility"
)

st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# 4. VaR Breaches
# -----------------------------
breaches = data[(data['Returns'].notna()) & (data['Returns'] < var_hist)]

fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x=data['Date'],
    y=data['Returns'],
    mode='lines',
    name="Returns",
    line=dict(color='blue')
))

fig4.add_trace(go.Scatter(
    x=breaches['Date'],
    y=breaches['Returns'],
    mode='markers',
    name="VaR Breaches",
    marker=dict(color='red', size=8, symbol='x')
))

fig4.update_layout(
    title="VaR Breaches",
    xaxis_title="Date",
    yaxis_title="Returns"
)

st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# 5. Price Prediction (Fixed)
# -----------------------------
data_pred = data.dropna().copy()

data_pred['Day'] = np.arange(len(data_pred))

X = data_pred[['Day']]
y = data_pred['close']

model = LinearRegression()
model.fit(X, y)

# Future prediction
future_days = np.arange(len(data_pred), len(data_pred) + 30).reshape(-1, 1)
pred = model.predict(future_days)

future_dates = pd.date_range(
    start=data_pred['Date'].iloc[-1] + pd.Timedelta(days=1),
    periods=30
)

fig5 = go.Figure()

# Actual prices
fig5.add_trace(go.Scatter(
    x=data_pred['Date'],
    y=y,
    mode='lines',
    name="Actual",
    line=dict(color='blue')
))

# Predictions
fig5.add_trace(go.Scatter(
    x=future_dates,
    y=pred,
    mode='lines',
    name="Prediction",
    line=dict(color='red', dash='dash')
))

fig5.update_layout(
    title="Price Prediction (Next 30 Days)",
    xaxis_title="Date",
    yaxis_title="Price (USD)"
)

st.plotly_chart(fig5, use_container_width=True)