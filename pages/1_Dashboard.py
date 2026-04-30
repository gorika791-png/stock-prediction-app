import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: #E6EDF3;
}
.block-container {
    padding-top: 2rem;
}
.stMetric {
    background-color: #161B22;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)
st.markdown("## 📈 StockSense AI")
st.caption("AI-powered stock prediction system")
# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Stock Dashboard", layout="wide")

st.title("📊 Apple Stock Dashboard")


# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("apple_dashboard_data.csv")

    # Fix date format (DD-MM-YYYY)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df.set_index('date', inplace=True)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    return df


df = load_data()
def card(title, value, subtitle=""):
    st.markdown(f"""
    <div style="
        background:#161B22;
        padding:20px;
        border-radius:12px;
        border:1px solid #30363d;
        margin-bottom:10px;">
        <h4 style="color:#8b949e;">{title}</h4>
        <h2 style="color:#E6EDF3;">{value}</h2>
        <p style="color:#6e7681;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# -------------------- DATE RANGE FILTER --------------------
st.sidebar.header("⚙️ Settings")

range_option = st.sidebar.selectbox(
    "Select Time Range",
    ["1 Month", "3 Months", "6 Months", "1 Year", "All"]
)

today = df.index.max()

if range_option == "1 Month":
    filtered_df = df[df.index > today - timedelta(days=30)]
elif range_option == "3 Months":
    filtered_df = df[df.index > today - timedelta(days=90)]
elif range_option == "6 Months":
    filtered_df = df[df.index > today - timedelta(days=180)]
elif range_option == "1 Year":
    filtered_df = df[df.index > today - timedelta(days=365)]
else:
    filtered_df = df

start_date = st.sidebar.date_input("Start Date", df.index.min())
end_date = st.sidebar.date_input("End Date", df.index.max())

filtered_df = df[(df.index >= pd.to_datetime(start_date)) &
                 (df.index <= pd.to_datetime(end_date))]

# -------------------- KPI SECTION --------------------
latest_price = filtered_df['close'].iloc[-1]
prev_price = filtered_df['close'].iloc[-2]

change = latest_price - prev_price
percent_change = (change / prev_price) * 100

high = filtered_df['high'].max()
low = filtered_df['low'].min()
avg_price = filtered_df['close'].mean()
volatility = filtered_df['close'].std()
filtered_df['ma20'] = filtered_df['close'].rolling(20).mean()
filtered_df['upper'] = filtered_df['ma20'] + 2 * filtered_df['close'].rolling(20).std()
filtered_df['lower'] = filtered_df['ma20'] - 2 * filtered_df['close'].rolling(20).std()

# col1, col2, col3, col4 = st.columns(4)
#
# col1.metric("💰 Current Price", f"${latest_price:.2f}", f"{percent_change:.2f}%")
# col2.metric("📈 Change", f"${change:.2f}")
# col3.metric("🔺 High", f"${high:.2f}")
# col4.metric("🔻 Low", f"${low:.2f}")
col1, col2, col3, col4 = st.columns(4)

with col1:
    card("💰 Price", f"${latest_price:.2f}", f"{percent_change:.2f}%")

with col2:
    card("📈 Change", f"${change:.2f}")

with col3:
    card("🔺 High", f"${high:.2f}")

with col4:
    card("🔻 Low", f"${low:.2f}")
col5, col6 = st.columns(2)
with col5:
    card("📊 Average Price", f"${avg_price:.2f}")
with col6:
    card("📉 Volatility", f"{volatility:.2f}")

# col5.metric("📊 Average Price", f"${avg_price:.2f}")
# col6.metric("📉 Volatility", f"{volatility:.2f}")

st.divider()

# -------------------- CANDLESTICK CHART --------------------
st.subheader("📊 Price Chart")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=filtered_df.index,
    open=filtered_df['open'],
    high=filtered_df['high'],
    low=filtered_df['low'],
    close=filtered_df['close'],
    name="Price"
))
fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['upper'], name='Upper Band'))
fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['lower'], name='Lower Band'))

# Add SMA if exists
if 'sma_50' in filtered_df.columns:
    fig.add_trace(go.Scatter(
        x=filtered_df.index,
        y=filtered_df['sma_50'],
        mode='lines',
        name='SMA 50'
    ))

# Add EMA if exists
# Create EMA if missing
if 'ema_20' not in filtered_df.columns:
    filtered_df['ema_20'] = filtered_df['close'].ewm(span=20).mean()

# Now safe to use
fig.add_trace(go.Scatter(
    x=filtered_df.index,
    y=filtered_df['ema_20'],
    mode='lines',
    name='EMA 20'
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, width='stretch')


# -------------------- VOLUME CHART --------------------
st.subheader("📊 Volume")

st.bar_chart(filtered_df['volume'])

# -------------------- INSIGHTS SECTION --------------------
st.subheader("🧠 Insights")

if percent_change > 0:
    st.success("📈 Stock is showing an upward trend today.")
else:
    st.error("📉 Stock is showing a downward trend today.")

# RSI insight (if exists)
if 'rsi_14' in filtered_df.columns:
    rsi = filtered_df['rsi_14'].iloc[-1]

    if rsi > 70:
        st.warning("⚠️ RSI indicates stock is OVERBOUGHT")
    elif rsi < 30:
        st.warning("⚠️ RSI indicates stock is OVERSOLD")
    else:
        st.info("ℹ️ RSI is in normal range")

# -------------------- DATA RANGE --------------------
st.divider()

st.write(f"📅 Data from {filtered_df.index.min().date()} to {filtered_df.index.max().date()}")

# add download button
st.download_button(
    label="📥 Download Data",
    data=filtered_df.to_csv().encode('utf-8'),
    file_name='stock_data.csv'
)
