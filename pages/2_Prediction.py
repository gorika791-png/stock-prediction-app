import streamlit as st
import pandas as pd
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Prediction", layout="wide")

# -------------------- CUSTOM UI --------------------
st.markdown("""
<style>
body {background-color:#0E1117;color:#E6EDF3;}
.block-container {padding-top:2rem;}
</style>
""", unsafe_allow_html=True)
st.markdown("## 📈 StockSense AI")
st.caption("AI-powered stock prediction system")
st.title("🤖 Stock Movement Prediction")

# -------------------- CARD FUNCTION --------------------
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

# -------------------- LOAD --------------------
df = pd.read_csv("cleaned_apple_stock.csv")
df = df.reset_index(drop=True)

model = joblib.load("model.pkl")
lr_model = joblib.load("lr_model.pkl")
rf_model = joblib.load("rf_model.pkl")

# -------------------- FEATURES --------------------
features = [
    'close','volume','sma_50','ema_20','ema_50','rsi_14',
    'macd','macd_signal','macd_histogram','bb_width',
    'volatility_20d','close_lag_1','close_lag_5',
    'volume_lag_1','rsi_signal','macd_signal_cross',
    'momentum','trend_strength','price_change'
]

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Input Selection")

mode = st.sidebar.radio("Select Mode", ["Latest Data", "Manual Selection"])

if mode == "Latest Data":
    input_data = df[features].tail(1)
else:
    index = st.sidebar.slider("Select Record", 0, len(df)-1, len(df)-1)
    input_data = df.loc[[index], features]
  # -------------------- BUTTON (GLOBAL) --------------------
col1, col2 = st.columns([4,1])

with col1:
    st.subheader("🔍 Run Prediction")

with col2:
    predict_clicked = st.button("🔮 Predict", width='stretch')

st.divider()

# -------------------- RUN PREDICTION ONCE --------------------
if predict_clicked:
    final_pred = model.predict(input_data)[0]

    lr_pred = lr_model.predict(input_data)[0]
    rf_pred = rf_model.predict(input_data)[0]

    lr_prob = lr_model.predict_proba(input_data)[0][1]
    rf_prob = rf_model.predict_proba(input_data)[0][1]

    avg_conf = (lr_prob + rf_prob) / 2
else:
    final_pred = None
    lr_pred = rf_pred = None
    lr_prob = rf_prob = avg_conf = 0

# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs(["📊 Result", "📂 Data", "🧠 Insights"])
with tab1:
    st.subheader("📊 Prediction Result")

    if final_pred is None:
        st.info("👉 Click Predict to see results")
    else:
        if final_pred == 1:
            st.success("📈 Stock likely to go UP")
        else:
            st.error("📉 Stock likely to go DOWN")

        st.progress(int(avg_conf * 100))
        st.caption(f"Confidence: {avg_conf*100:.2f}%")

        # Model Comparison
        st.subheader("🤖 Model Comparison")

        col1, col2 = st.columns(2)

        with col1:
            card("Logistic Regression",
                 "UP" if lr_pred==1 else "DOWN",
                 f"{lr_prob*100:.2f}%")

        with col2:
            card("Random Forest",
                 "UP" if rf_pred==1 else "DOWN",
                 f"{rf_prob*100:.2f}%")

        # Recommendation
        if lr_pred == rf_pred:
            if avg_conf > 0.7:
                st.success("📈 Strong BUY signal")
            elif avg_conf > 0.55:
                st.info("👍 Moderate BUY signal")
            else:
                st.warning("⚠️ Weak signal")
        else:
            st.error("❌ HOLD – Models disagree")
with tab2:
    st.subheader("📂 Input Data")

    st.dataframe(input_data)

    st.metric("💰 Current Price", f"${df['close'].iloc[-1]:.2f}")
with tab3:
    st.subheader("🧠 Insights")

    st.info("""
    - Prediction based on RSI, MACD, Moving Averages
    - Random Forest captures complex patterns
    - Logistic Regression provides baseline
    """)

    st.warning("""
    ⚠️ This prediction is based on historical data and may not reflect real-time market behavior.
    """)
  
