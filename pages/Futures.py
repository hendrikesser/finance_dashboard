import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.optimize import fsolve


# ===========================
# Page Title & Theme
# ===========================
st.set_page_config(page_title="Futures Dashboard", layout="wide")
st.title("📈 Futures Dashboard")

section = st.selectbox(
    "Select subsection:", 
    ["Basics, Pay Offs and Strategies", "Commodity and Stock Futures", "Foreign Exchange and Interest Rate Futures", "Index and VIX Futures"]
)

st.write("coming soon...")

