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
# Subsection Selection
# ===========================
section = st.selectbox(
    "Select subsection:", 
    ["Foreign Exchange Swap", "Interest Rate Swap", "Credit Default Swap"]
)



st.write("coming soon...")


