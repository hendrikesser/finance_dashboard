
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ===========================
# Page Title & Theme
# ===========================
st.set_page_config(page_title="Options Dashboard", layout="wide")
st.title("📈 Options Dashboard")


# ===========================
# Custom CSS for blue dropdown text
# ===========================
st.markdown("""
<style>
div[data-baseweb="select"] > div {
    color: #1a73e8 !important;
    font-size: 16px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ===========================
# Section Navigation
# ===========================
section = st.selectbox(
    "Select Subsection:",
    [
        "Basics & Put-Call Parity",
        "Binomial Tree Model",
        "Black-Scholes Model",
        "Implied Volatility",
        "Greeks & Risk Management",
    ])
# =======================================
# SHARED FUNCTIONS
# =======================================
st.write("coming soon...")