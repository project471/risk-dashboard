import streamlit as st
import requests
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
# Fetch Data (Twelve Data ONLY)
# -----------------------------
@st.cache_data(ttl=300)
def fetch_data(symbol):

    API_KEY = "a8f08ca485cb4066ba4ab47da49514fb"

    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=200&apikey={API_KEY}"

    r = requests.get(url).json()

    if "values" not in r:
        return None

    df = pd.DataFrame(r["values"])

    df = df.rename(columns={
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "close"
    })

    df["Date"] = pd.to_datetime(df["Date"])

    # Safe numeric conversion
    for col in ["Open", "High", "Low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# -----------------------------
# Load Data
# -----------------------------
data = fetch_data(ticker)

if data is None or data.empty:
    st.error("Invalid ticker or API limit reached.")
    st.stop()

# -----------------------------
# Returns & Risk Metrics
# -----------------------------
data['Returns'] = data['close'].pct_change()
returns = data['Returns'].dropna()

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
# 1. Candlestick Chart
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
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# 2. Returns Distribution
# -----------------------------
fig2 = go.Figure()

fig2.add_trace(go.Histogram(
    x=returns,
    nbinsx=50,
    marker_color='blue'
))

fig2.add_vline(x=var_hist, line_color="red")

fig2.update_layout(title="Returns Distribution with VaR")

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 3. Rolling Volatility
# -----------------------------
fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=data['Date'],
    y=data['Volatility'],
    fill='tozeroy',
    line=dict(color='purple')
))

fig3.update_layout(title="30-Day Rolling Volatility")

st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# 4. Price Prediction
# -----------------------------
data_pred = data.dropna().copy()
data_pred['Day'] = np.arange(len(data_pred))

X = data_pred[['Day']]
y = data_pred['close']

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(len(data_pred), len(data_pred) + 30).reshape(-1, 1)
pred = model.predict(future_days)

future_dates = pd.date_range(
    start=data_pred['Date'].max() + pd.Timedelta(days=1),
    periods=30
)

fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x=data_pred['Date'],
    y=y,
    mode='lines',
    name="Actual",
    line=dict(color='black')
))

fig4.add_trace(go.Scatter(
    x=future_dates,
    y=pred,
    mode='lines',
    name="Prediction",
    line=dict(color='purple', dash='dash')
))

fig4.update_layout(title="Price Prediction (Next 30 Days)")

st.plotly_chart(fig4, use_container_width=True)