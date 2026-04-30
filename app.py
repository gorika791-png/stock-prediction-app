
import streamlit as st

st.set_page_config(page_title="Apple Stock App", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.info("Select a page below:")

# Main page
st.title("Apple Inc Stock Prediction App")

st.markdown("""
Welcome to the Stock Prediction System 🚀

### 📌 Features:
- 📊 Dashboard (Graphs & Indicators)
- 🤖 Prediction (ML Model)
- 📈 Model Performance


👉 Use the sidebar to navigate between pages.
""")