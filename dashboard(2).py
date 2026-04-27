import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Risk Dashboard", layout="wide")
st.title("📊 Risk Management Dashboard (VaR, CVaR & Prediction)")

# -----------------------------
# Inputs
# -----------------------------
ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95)

# -----------------------------
# Fetch Data (FINNHUB ONLY)
# -----------------------------
@st.cache_data(ttl=300)
def fetch_data(symbol):

    API_KEY = "d7nlp79r01qm3636uragd7nlp79r01qm3636urb0"

    try:
        end = int(time.time())
        start = end - (60 * 60 * 24 * 180)  # last 6 months

        url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&from={start}&to={end}&token={API_KEY}"
        response = requests.get(url)
        data = response.json()

        if data.get("s") != "ok":
            return None

        df = pd.DataFrame({
            "Date": pd.to_datetime(data["t"], unit='s'),
            "Open": data["o"],
            "High": data["h"],
            "Low": data["l"],
            "close": data["c"]
        })

        df = df.sort_values("Date").reset_index(drop=True)

        return df

    except:
        return None

# -----------------------------
# Load Data (prevent reruns)
# -----------------------------
if "data" not in st.session_state or st.session_state.get("ticker") != ticker:
    st.session_state["data"] = fetch_data(ticker)
    st.session_state["ticker"] = ticker

data = st.session_state["data"]

if data is None or data.empty:
    st.error("Invalid ticker or API issue (Finnhub).")
    st.stop()

# -----------------------------
# Returns & Risk Metrics
# -----------------------------
data["Returns"] = data["close"].pct_change()
returns = data["Returns"].dropna()

if len(returns) == 0:
    st.error("Not enough data.")
    st.stop()

var_hist = np.percentile(returns, (1 - confidence_level) * 100)
var_param = norm.ppf(1 - confidence_level, returns.mean(), returns.std())
cvar = returns[returns <= var_hist].mean()

data["Volatility"] = data["Returns"].rolling(30).std()

# -----------------------------
# Metrics
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Historical VaR", round(var_hist, 6))
col2.metric("Parametric VaR", round(var_param, 6))
col3.metric("CVaR", round(cvar, 6))

# -----------------------------
# Candlestick Chart
# -----------------------------
fig1 = go.Figure()

fig1.add_trace(go.Candlestick(
    x=data["Date"],
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["close"],
    increasing_line_color="green",
    decreasing_line_color="red"
))

fig1.update_layout(
    title="Stock Price (Candlestick)",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# Returns Distribution
# -----------------------------
fig2 = go.Figure()

fig2.add_trace(go.Histogram(
    x=returns,
    nbinsx=50,
    marker_color="blue"
))

fig2.add_vline(
    x=var_hist,
    line_color="red",
    annotation_text="VaR"
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Rolling Volatility
# -----------------------------
fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=data["Date"],
    y=data["Volatility"],
    fill="tozeroy",
    line=dict(color="purple")
))

st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# VaR Breaches
# -----------------------------
breaches = data[data["Returns"] < var_hist]

fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x=data["Date"],
    y=data["Returns"],
    name="Returns"
))

fig4.add_trace(go.Scatter(
    x=breaches["Date"],
    y=breaches["Returns"],
    mode="markers",
    marker=dict(color="red", size=8),
    name="Breaches"
))

st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# Price Prediction
# -----------------------------
data_pred = data.dropna().copy()
data_pred["Day"] = np.arange(len(data_pred))

X = data_pred[["Day"]]
y = data_pred["close"]

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(len(data_pred), len(data_pred) + 30).reshape(-1, 1)
pred = model.predict(future_days)

future_dates = pd.date_range(
    start=data_pred["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=30
)

fig5 = go.Figure()

fig5.add_trace(go.Scatter(
    x=data_pred["Date"],
    y=y,
    name="Actual",
    line=dict(color="black")
))

fig5.add_trace(go.Scatter(
    x=future_dates,
    y=pred,
    name="Prediction",
    line=dict(color="purple", dash="dash")
))

st.plotly_chart(fig5, use_container_width=True)