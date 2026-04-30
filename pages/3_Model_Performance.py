import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, precision_score,
    recall_score, f1_score
)

# ---------------- UI ----------------
st.set_page_config(page_title="Model Performance", layout="wide")

st.markdown("## 📈 StockSense AI")
st.caption("AI-powered stock prediction system")
st.title("📊 Model Performance Analysis")

# ---------------- LOAD ----------------
df = pd.read_csv("cleaned_apple_stock.csv")

features = [
    'close','volume','sma_50','ema_20','ema_50','rsi_14',
    'macd','macd_signal','macd_histogram','bb_width',
    'volatility_20d','close_lag_1','close_lag_5',
    'volume_lag_1','rsi_signal','macd_signal_cross',
    'momentum','trend_strength','price_change'
]

X = df[features]
y = df['target_direction']

lr_model = joblib.load("lr_model.pkl")
rf_model = joblib.load("rf_model.pkl")

lr_pred = lr_model.predict(X)
rf_pred = rf_model.predict(X)

# ---------------- METRICS ----------------
lr_acc = accuracy_score(y, lr_pred)
rf_acc = accuracy_score(y, rf_pred)

rf_precision = precision_score(y, rf_pred)
rf_recall = recall_score(y, rf_pred)
rf_f1 = f1_score(y, rf_pred)

cm = confusion_matrix(y, rf_pred)

report = classification_report(y, rf_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "📈 Metrics", "🔍 Confusion Matrix", "📄 Report & Insights"
])

# ================= OVERVIEW =================
with tab1:
    st.subheader("📊 Accuracy Comparison")

    col1, col2 = st.columns(2)

    col1.metric("Logistic Regression", f"{lr_acc*100:.2f}%")
    col2.metric("Random Forest", f"{rf_acc*100:.2f}%")

    # Best model
    st.subheader("🏆 Best Model")

    if rf_acc > lr_acc:
        st.success("Random Forest performs better")
    else:
        st.success("Logistic Regression performs better")

    if rf_acc > 0.90:
        st.warning("⚠️ High accuracy detected → possible overfitting")

# ================= METRICS =================
with tab2:
    st.subheader("📈 Advanced Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Precision", f"{rf_precision*100:.2f}%")
    col2.metric("Recall", f"{rf_recall*100:.2f}%")
    col3.metric("F1 Score", f"{rf_f1*100:.2f}%")

    st.info("""
    ✔ Precision → How accurate positive predictions are  
    ✔ Recall → How well model captures actual positives  
    ✔ F1 Score → Balance of precision & recall  
    """)

# ================= CONFUSION MATRIX =================
with tab3:
    st.subheader("🔍 Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Down', 'Up'],
        yticklabels=['Down', 'Up']
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    st.pyplot(fig)

    st.info("""
    📊 Interpretation:
    - True Positives → Correct UP predictions  
    - True Negatives → Correct DOWN predictions  
    - False Positives → Risky BUY signals  
    - False Negatives → Missed opportunities  
    """)

# ================= REPORT =================
with tab4:
    st.subheader("📄 Classification Report")

    st.dataframe(
        report_df.style.format("{:.2f}"),
        width='stretch'
    )

    st.subheader("📊 Insights")

    st.success("""
    ✔ Random Forest captures complex patterns  
    ✔ Logistic Regression acts as baseline  
    ✔ Ensemble comparison improves reliability  
    """)

    st.subheader("📌 Final Conclusion")

    if rf_acc > lr_acc:
        st.success("""
        ✅ Random Forest is the best model

        ✔ Higher accuracy  
        ✔ Better pattern recognition  
        ✔ More reliable predictions  
        """)

# ---------------- FOOTER ----------------
st.warning("⚠️ Based on historical data. Real market may differ.")
