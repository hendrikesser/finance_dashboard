import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===========================
# Page Title & Layout
# ===========================
st.set_page_config(
    page_title="Derivatives Dashboard – Home",
    layout="wide",
)

# ===========================
# Custom CSS for Consistent Styling
# ===========================
st.markdown("""
<style>
div[data-baseweb="select"] > div {
    color: #1a73e8 !important;
    font-size: 16px !important;
    font-weight: bold;
}
.big-title {
    font-size: 42px !important;
    font-weight: 700 !important;
    color: #004c99 !important;
}
.section-card {
    padding: 25px;
    border-radius: 15px;
    background-color: #f7faff;
    border: 1px solid #d8e6ff;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)


# ===========================
# Title
# ===========================
st.markdown("<div class='big-title'>📊 Financial Dashboard </div>", unsafe_allow_html=True)

st.write("""
Welcome to the **Financial Dashboard**, an interactive learning platform covering the four major building blocks of modern financial markets:

### **💵 Bonds • 📈 Futures • 📘 Options • 🔄 Swaps**""")
st.write("Each section combines:")

col1, col2 = st.columns(2)
with col1: 
    st.write("""
- Intuitive explanations  
- Interactive simulations  
- Payoff diagrams""")
with col2: 
    st.write("""
- Pricing models  
- Real numerical examples  
- Hedging applications""")

st.write("""
Use the sidebar to navigate between modules.""")

st.markdown("---")


# ===========================
# Section Cards
# ===========================
st.header("📂 Dashboard Sections")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)



# ----- BONDS CARD -----
with col1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("💵 Bonds")
    st.write("""
Understand interest-bearing securities:
- Zero-coupon & coupon-paying bonds  
- Yield curves  
- Macauly and Modified Duration 
- Price–yield dynamics  
""")
    st.write("➡️ Gain intuition for fixed-income pricing and risk.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----- FUTURES CARD -----
with col2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📈 Futures")
    st.write("""
Explore the mechanics of futures contracts:
- Long/short payoffs  
- Mark-to-market (MtM), Basis & Rollover Risk
- Hedging with underlying exposure  
- Commodity, equity, FX, interest rate, and VIX futures   
""")
    st.write("➡️ Ideal starting point to understand linear derivatives.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----- OPTIONS CARD -----
with col3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📘 Options")
    st.write("""
Explore nonlinear derivatives:
- Calls / puts and Payoff diagrams  
- Binomial pricing  and Black–Scholes model  
- Greeks (Delta, Gamma, Vega, Theta, Rho)  
- Volatility smiles & implied volatility  
""")
    st.write("➡️ Perfect for understanding asymmetric payoffs and risk sensitivities.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----- SWAPS CARD -----
with col4:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🔄 Swaps")
    st.write("""
Dive into the world of swaps:
- Interest rate swaps (IRS)  
- Fixed vs floating legs  
- FX Swaps
- Credit Default Swaps 
""")
    st.write("➡️ Learn how institutions reshape risk exposures through swaps.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ===========================
# How to Use
# ===========================
st.header("🎯 How to Use This Dashboard")

st.write("""
Use the **sidebar** to navigate between modules.

Each module includes:
- Theory explained intuitively  
- Sliders to adjust market parameters  
- Interactive charts  
- Numerical calculations  
- Realistic hedging examples  

""")
