import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.optimize import fsolve


# ===========================
# Page Title & Theme
# ===========================
st.set_page_config(page_title="Swaps Dashboard", layout="wide")
st.title("📈 Swaps Dashboard")

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
# Subsection Selection
# ===========================
section = st.selectbox(
    "Select subsection:", 
    ["Foreign Exchange Swap", "Interest Rate Swap", "Credit Default Swap"]
)



st.write("coming soon...")


