import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm

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
# SHARED FUNCTIONS (Option Math)
# =======================================

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Standard Black-Scholes Formula"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price, d1, d2

def bsm_greeks(S, K, T, r, sigma):
    """Calculate the main Greeks"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100 # per 1% vol change
    
    return {"Delta Call": delta_call, "Delta Put": delta_put, "Gamma": gamma, "Vega": vega}

# =======================================
# 1 Basics & Put-Call Parity
# =======================================

if section == "Basics & Put-Call Parity":

    # --- Theoretical Introduction ---
    st.header("📘 Options Theory & Strategy Builder")

    st.markdown("""
    ### 🏗️ Introduction to Options
    An **option** is a non-binding agreement—a right, but not an obligation—to trade an underlying asset at a pre-determined **Strike Price (K)** at or before expiration. 
    Unlike binding forward contracts which have zero value at initiation, options always have a **positive value** because they offer protection against adverse price moves, requiring an upfront **premium**.

    ### 📉 Basic Payoff Structures
    The value of an option at expiration ($T$) depends on the spot price ($S_T$) relative to the strike ($K$):""")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Call Option** — *Right to Buy*

            **Moneyness**
            - **In-the-Money (ITM):** $( S_T > K )$
            - **At-the-Money (ATM):** $( S_T = K )$
            - **Out-of-the-Money (OTM):** $( S_T < K )$

            Payoff = $\max(S_T - K, 0)$
            """)

    with col2:
        st.markdown(
            """
            **Put Option** — *Right to Sell*

            **Moneyness**
            - **In-the-Money (ITM):** $( S_T < K )$
            - **At-the-Money (ATM):** $( S_T = K )$
            - **Out-of-the-Money (OTM):** $( S_T > K )$

            Payoff = $\max(K - S_T, 0)$
            """)

    st.markdown("---")
    # --- Sidebar Parameters ---
    st.sidebar.header("Put_-Call Parity Inputs")
    S_T = st.sidebar.slider("Current Stock Price (S)", 10.0, 500.0, 140.0)
    K_strike = st.sidebar.slider("Strike Price (K)", 10.0, 500.0, 140.0)
    r_rate = st.sidebar.slider("Risk-Free Rate (r)", 0.0, 0.20, 0.05)
    div_yield = st.sidebar.slider("Dividend Yield (y)", 0.0, 0.20, 0.02)
    T_days = st.sidebar.slider("Days to Maturity (T)", 1, 365, 30)

    # Converting days to annual years for formulas
    T_years = T_days / 365.0
    
    # --- Theoretical Factors ---
    st.subheader("⚙️ Factors Influencing Option Prices")
    st.write("Option value increases if the probability or the size of the payoff increases.")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("**1. Volatility (σ) (+/+)**")
        st.write("Increases both Call/Put value. Payoffs are convex; higher dispersion leads to higher potential gains while losses are capped.")
        st.markdown("**2. Dividends (D) (-/+)**")
        st.write("Decreases Call value and increases Put value as the stock price typically falls on ex-dividend dates.")
    with col_f2:
        st.markdown("**3. Interest Rates (r) (+/-)**")
        st.write("Increases Call value (delaying payment) and decreases Put value (delaying income).")
        st.markdown("**4. Time (T-t) (+/+)**")
        st.write("For American options, longer maturity never hurts. For European, it generally increases value due to higher dispersion.")

    st.markdown("---")

    # --- Strategy & Portfolio Builder ---
    st.subheader("🛠️ Strategy & Portfolio Builder")
    st.write("Combine positions to build a portfolio (e.g., Buy 1 Call + Buy 1 Put = **Straddle**).")

    # Interactive Strategy Inputs
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        pos_stock = st.number_input("Shares of Stock", value=0, key="stock_pos")
        S_curr = st.number_input("Current Spot Price (S)", value=100.0)
    with col_s2:
        pos_call = st.number_input("Long/Short Calls", value=0, key="call_pos")
        K_call = st.number_input("Call Strike Price", value=100.0, key="call_k")
        C_prem = st.number_input("Call Premium", value=5.0, key="call_p")
    with col_s3:
        pos_put = st.number_input("Long/Short Puts", value=0, key="put_pos")
        K_put = st.number_input("Put Strike Price", value=100.0, key="put_k")
        P_prem = st.number_input("Put Premium", value=5.0, key="put_p")

    # Calculation logic
    s_range = np.linspace(S_curr * 0.5, S_curr * 1.5, 100)
    stock_payoff = pos_stock * (s_range - S_curr)
    call_payoff = pos_call * (np.maximum(s_range - K_call, 0) - C_prem)
    put_payoff = pos_put * (np.maximum(K_put - s_range, 0) - P_prem)
    total_payoff = stock_payoff + call_payoff + put_payoff

    # --- Payoff Visualization ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_range, y=total_payoff, name='Total Portfolio', line=dict(color='white', width=4)))
    if pos_call != 0: fig.add_trace(go.Scatter(x=s_range, y=call_payoff, name='Call Component', line=dict(dash='dash')))
    if pos_put != 0: fig.add_trace(go.Scatter(x=s_range, y=put_payoff, name='Put Component', line=dict(dash='dash')))
    
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(title="Portfolio Profit/Loss at Expiration", xaxis_title="Stock Price at T", yaxis_title="Profit / Loss")
    st.plotly_chart(fig)

    st.subheader("⚖️ Theoretical Price Bounds")
    # Pre-calculating Present Value of Strike
    pv_k_call = K_call * np.exp(-r_rate * T_years)
    pv_k_put = K_put * np.exp(-r_rate * T_years)

    # Calculations based on Day 8, Slides 18-19
    c_lower = max(0.0, S_curr - pv_k_call)
    p_lower = max(0.0, pv_k_put - S_curr)

    col_b1, col_b2 = st.columns(2)
    col_b1, col_b2 = st.columns(2)

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.write("**Call Bounds**")

        st.info(
            f"{c_lower:.2f} ≤ Cₜ ≤ {S_curr:.2f}"
        )

        st.markdown(
            """
            **Interpretation**
            - Below lower bound → **Arbitrage opportunity**
            - Above upper bound → **Dominance violation** (call > stock)
            """
        )

        if pos_call != 0 and C_prem < c_lower:
            st.error("🚨 Arbitrage Alert: Call price is below its lower bound!")

        if pos_call != 0 and C_prem > S_curr:
            st.error("🚨 Arbitrage Alert: Call price is above its upper bound!")


    with col_b2:
        st.write("**Put Bounds**")

        st.info(
            f"{p_lower:.2f} ≤ Pₜ ≤ {pv_k_put:.2f}"
        )

        st.markdown(
            """
            **Interpretation**
            - Below lower bound → **Arbitrage opportunity**
            - Above upper bound → **Dominance violation** (put > PV(K))
            """
        )

        if pos_put != 0 and P_prem < p_lower:
            st.error("🚨 Arbitrage Alert: Put price is below its lower bound!")

        if pos_put != 0 and P_prem > pv_k_put:
            st.error("🚨 Arbitrage Alert: Put price is above its upper bound!")



    st.subheader("🎯 Current Moneyness Status")

    def check_moneyness(s, k, is_call=True):
        if abs(s - k) < (s * 0.01): # 1% threshold for ATM
            return "At-the-Money (ATM)", "gray"
        if is_call:
            return ("In-the-Money (ITM)", "green") if s > k else ("Out-of-the-Money (OTM)", "red")
        else:
            return ("In-the-Money (ITM)", "green") if s < k else ("Out-of-the-Money (OTM)", "red")
    m_col1, m_col2 = st.columns(2)

    if pos_call != 0:
            status, color = check_moneyness(S_curr, K_call, True)
            m_col1.markdown(f"**Call Strike ({K_call:.2f}):** :{color}[{status}]")

    if pos_put != 0:
            status, color = check_moneyness(S_curr, K_put, False)
            m_col2.markdown(f"**Put Strike ({K_put:.2f}):** :{color}[{status}]")
            
    # ----------------------------
    # Pre-defined Strategy Theory
    # ----------------------------
    st.markdown("---")
    st.write("**Popular Portfolio Strategies**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        - **Covered Call**  
        Long 1 stock + short 1 call  
        → Income strategy, sacrifices upside""")
    with col2:
        st.markdown("""
        - **Protected Short**  
        Short 1 stock + long 1 call  
        → Caps upside risk of a short position""")
    with col3:
        st.markdown("""
        - **Straddle**  
        Long 1 call + long 1 put (same strike)  
        → Pure volatility bet, direction-neutral
        """
    )

    # --- Put-Call Parity  ---
    st.markdown("---")
    st.subheader("⚖️ Put-Call Parity & No-Arbitrage Theory")
    
    st.write("""
    **Theory:** Put-Call Parity (PCP) is a fundamental relationship that must hold to prevent arbitrage. 
    It is derived by creating a **Synthetic Forward**: if you buy a call and sell a put with the same strike ($K$) 
    and maturity ($T$), your net payoff at expiration is exactly $S_T - K$. 
    """)
    
    st.latex(r"C_t - P_t = S_t e^{-q(T-t)} - K e^{-r(T-t)}")
    
    st.info("""
    **No-Arbitrage Logic:**
    - If $C - P > S e^{-qT} - K e^{-rT}$, the call is relatively overpriced. **Strategy:** Sell the synthetic forward (Short Call + Long Put) and buy the underlying.
    - If $C - P < S e^{-qT} - K e^{-rT}$, the put is relatively overpriced. **Strategy:** Buy the synthetic forward (Long Call + Short Put) and sell the underlying.
    """)

    # --- Theoretical Price Bounds ---
    st.subheader("📏 Theoretical Price Bounds")
    st.write("""
    Even without a complex model like Black-Scholes, we can determine the 'rational' range for option prices 
    based on the principle that an option cannot be worth less than its immediate exercise value or more than the underlying.
    """)
    

    # --- Theoretical Calculator ---
    st.markdown("### 🧮 Parity & Fair Value Calculator")
    st.write("Compare market premiums against theoretical parity to find mispricings.")

    # Calculation of Present Values (Day 8, Page 7)
    pv_k = K_strike * np.exp(-r_rate * T_years)
    pv_s_div = S_T * np.exp(-div_yield * T_years) 

    calc_col1, calc_col2 = st.columns(2)

    with calc_col1:
        # Users input the current MARKET prices for comparison
        C_mkt = st.number_input("Market Call Price", value=10.0)
        # Fair Put = C - S*e^(-yT) + K*e^(-rT)
        p_fair = C_mkt - pv_s_div + pv_k
        st.metric("Fair Put Price", f"{max(0.0, p_fair):.2f}")

    with calc_col2:
        P_mkt = st.number_input("Market Put Price", value=5.0)
        # Fair Call = P + S*e^(-yT) - K*e^(-rT)
        c_fair = P_mkt + pv_s_div - pv_k
        st.metric("Fair Call Price", f"{max(0.0, c_fair):.2f}")

    # --- Explicit Arbitrage Steps (Day 8, Page 9-11) ---
    st.markdown("### 💸 Arbitrage Execution Steps")

    # PCP Logic: C + PV(K) should equal P + PV(S_adjusted)
    lhs = C_mkt + pv_k
    rhs = P_mkt + pv_s_div
    diff = lhs - rhs

    if abs(diff) < 0.01:
        st.success("The market is in parity. No arbitrage opportunity detected.")
    else:
        if diff > 0:
            # Scenario A: Call side is too expensive [cite: 513, 514]
            st.error(f"Arbitrage: Call side is Overvalued by {abs(diff):.2f}")
            st.write("**Execution Steps (Sell High, Buy Low):**")
            st.write(f"1. **Short 1 Call**: Receive ${C_mkt:.2f} premium.")
            st.write(f"2. **Long 1 Put**: Pay ${P_mkt:.2f} premium.")
            st.write(f"3. **Buy {np.exp(-div_yield * T_years):.4f} units of Stock**: Pay ${pv_s_div:.2f}.")
            st.write(f"4. **Lend ${pv_k:.2f}**: Invest at risk-free rate until expiration.")
            st.info(f"**Instant Risk-Free Profit:** ${abs(diff):.2f}")
        else:
            # Scenario B: Put side is too expensive [cite: 522]
            st.error(f"Arbitrage: Put side is Overvalued by {abs(diff):.2f}")
            st.write("**Execution Steps (Sell High, Buy Low):**")
            st.write(f"1. **Long 1 Call**: Pay ${C_mkt:.2f} premium.")
            st.write(f"2. **Short 1 Put**: Receive ${P_mkt:.2f} premium.")
            st.write(f"3. **Short {np.exp(-div_yield * T_years):.4f} units of Stock**: Receive ${pv_s_div:.2f}.")
            st.write(f"4. **Borrow ${pv_k:.2f}**: Finance the position at the risk-free rate.")
            st.info(f"**Instant Risk-Free Profit:** ${abs(diff):.2f}")

    # --- Theoretical Footnote ---
    with st.expander("📝 How does this work?"):
        st.write("""
        The steps above are based on the **Synthetic Forward** relationship. 
        Because $C - P = S e^{-qT} - K e^{-rT}$, any deviation creates a 'money machine'. 
        - If the Call is too expensive, you sell the 'synthetic' version and buy the real one.
        - If the Put is too expensive, you sell the Put/Call combo that replicates a short stock position.
        """)

    # --- Advanced Properties  ---
    with st.expander("📚 Deep Dive: Early Exercise & Style Properties"):
        st.write("""
        **Why do dividends matter for American Options?**
        It is **never optimal** to exercise an American Call early on a non-dividend paying stock 
        because you lose the 'insurance' value and the time value of money on the strike.
        
        However, with **Dividends (D)**:
        - **Calls:** High dividends make early exercise *more* likely (to capture the dividend).
        - **Puts:** High interest rates and low dividends make early exercise *more* likely (to receive the strike cash sooner).
        """)
        
        st.latex(r"C_{Amer} \geq C_{Euro} \quad \text{and} \quad P_{Amer} \geq P_{Euro}")
        st.caption("Because you have more rights (exercise any time), an American option must be worth at least as much as a European one.")

# =======================================
# 2️⃣ BINOMIAL TREE MODEL
# =======================================
elif section == "Binomial Tree Model":
    st.header("🌳 Binomial Options Pricing (1-Step)")
    
    st.sidebar.header("Binomial Inputs")
    S0 = st.sidebar.number_input("Spot Price", value=100.0)
    u = st.sidebar.slider("Up Factor (u)", 1.01, 1.50, 1.10)
    d = st.sidebar.slider("Down Factor (d)", 0.50, 0.99, 0.90)
    r_b = st.sidebar.slider("Risk-free rate (r)", 0.0, 0.10, 0.05)
    K_b = st.sidebar.number_input("Strike Price", value=100.0)

    # Math
    q = (np.exp(r_b) - d) / (u - d) # Risk-neutral probability
    Su = S0 * u
    Sd = S0 * d
    Cu = max(Su - K_b, 0)
    Cd = max(Sd - K_b, 0)
    C0 = np.exp(-r_b) * (q * Cu + (1 - q) * Cd)

    st.write("""
    The Binomial Model discretizes price movements. In a one-step model, the stock can move to either an **Up** state or a **Down** state.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Risk-Neutral Probability ($q$)")
        st.latex(r"q = \frac{e^{r\Delta t} - d}{u - d}")
        st.write(f"Calculated $q$: **{q:.4f}**")
        
        st.write("#### Option Value at $t=0$")
        st.latex(r"f = e^{-r\Delta t} [q f_u + (1-q) f_d]")
        st.metric("Fair Call Price", f"{C0:.2f}")

    with col2:
        st.write("#### Visual Representation")
        # Creating a simple plot to mimic a tree
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot([0, 1], [S0, Su], marker='o', color='green', label='Up state')
        ax.plot([0, 1], [S0, Sd], marker='o', color='red', label='Down state')
        ax.annotate(f'S={Su}\nC={Cu}', (1, Su))
        ax.annotate(f'S={Sd}\nC={Cd}', (1, Sd))
        ax.annotate(f'S₀={S0}', (0, S0), textcoords="offset points", xytext=(-30,0))
        ax.set_title("1-Step Binomial Tree")
        ax.axis('off')
        st.pyplot(fig)

# =======================================
# 3️⃣ BLACK-SCHOLES MODEL
# =======================================
elif section == "Black-Scholes Model":
    st.header("🧪 Black-Scholes-Merton (BSM) Model")
    
    st.sidebar.header("BSM Inputs")
    S = st.sidebar.number_input("Spot Price", value=100.0)
    K = st.sidebar.number_input("Strike Price", value=105.0)
    T = st.sidebar.slider("Maturity (T)", 0.01, 5.0, 1.0)
    r = st.sidebar.slider("Risk-free Rate", 0.0, 0.15, 0.05)
    sigma = st.sidebar.slider("Volatility (σ)", 0.05, 1.0, 0.20)

    call_p, d1, d2 = black_scholes(S, K, T, r, sigma, "call")
    put_p, _, _ = black_scholes(S, K, T, r, sigma, "put")

    st.write("The BSM model assumes stock prices follow a Geometric Brownian Motion with constant volatility.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Call Price", f"${call_p:.2f}")
    c2.metric("Put Price", f"${put_p:.2f}")
    c3.metric("d1", f"{d1:.4f}")

    st.markdown("---")
    st.subheader("📊 Price Sensitivity to Volatility")
    vols = np.linspace(0.05, 0.8, 50)
    prices = [black_scholes(S, K, T, r, v, "call")[0] for v in vols]
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(vols*100, prices, color='#1a73e8', lw=3)
    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Call Price ($)")
    ax.set_title("Option Price vs Volatility")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# =======================================
# 4️⃣ IMPLIED VOLATILITY
# =======================================
elif section == "Implied Volatility":
    st.header("📉 Implied Volatility (IV)")
    st.write("""
    **Implied Volatility** is the σ value that, when plugged into the Black-Scholes formula, 
    makes the theoretical price equal to the **market price**. 
    It represents the market's expectation of future risk.
    """)
    
    market_price = st.number_input("Enter Market Call Price", value=10.0)
    st.info("In practice, IV is found using numerical methods like Newton-Raphson.")
    
    # Placeholder for actual IV solver
    st.write("#### 🔹 The Volatility Smile")
    st.write("Usually, IV is not constant across all strikes. This 'smile' or 'skew' indicates that the market prices tail-risks differently.")
    
    # Dummy Vol Smile Plot
    strikes = np.linspace(80, 120, 10)
    iv_smile = [0.25, 0.22, 0.20, 0.19, 0.18, 0.19, 0.21, 0.23, 0.26, 0.30]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(strikes, iv_smile, marker='o', linestyle='--')
    ax.set_title("Example of a Volatility Smile")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Implied Volatility")
    st.pyplot(fig)

# =======================================
# 5️⃣ GREEKS & RISK MANAGEMENT
# =======================================
elif section == "Greeks & Risk Management":
    st.header("🛡️ The Greeks")
    st.write("Greeks measure the sensitivity of the option price to various parameters.")

    # Sidebar for Greeks
    st.sidebar.header("Parameters")
    S_g = st.sidebar.number_input("Spot Price", value=100.0)
    K_g = st.sidebar.number_input("Strike Price", value=100.0)
    T_g = st.sidebar.slider("Time", 0.1, 2.0, 0.5)
    sigma_g = st.sidebar.slider("Vol", 0.1, 0.5, 0.2)
    r_g = 0.05

    greeks = bsm_greeks(S_g, K_g, T_g, r_g, sigma_g)

    cols = st.columns(4)
    cols[0].metric("Delta (Call)", f"{greeks['Delta Call']:.3f}")
    cols[1].metric("Gamma", f"{greeks['Gamma']:.4f}")
    cols[2].metric("Vega", f"{greeks['Vega']:.3f}")
    cols[3].metric("Delta (Put)", f"{greeks['Delta Put']:.3f}")

    st.markdown("---")
    
    # Explanations
    st.subheader("🔍 Greek Definitions")
    st.markdown("""
    * **Delta ($\Delta$):** Sensitivity to the underlying price. $\Delta = 0.5$ means for every \$1 move in the stock, the option moves \$0.50.
    * **Gamma ($\Gamma$):** The rate of change in Delta. Measures the 'acceleration' of the price.
    * **Vega:** Sensitivity to volatility. High vega means the option is sensitive to changes in market fear.
    * **Theta ($\Theta$):** Sensitivity to time decay. Options lose value as they approach expiry (all else equal).
    """)

    # Delta across Spot Prices
    spots = np.linspace(S_g * 0.5, S_g * 1.5, 100)
    deltas = [bsm_greeks(s, K_g, T_g, r_g, sigma_g)["Delta Call"] for s in spots]
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(spots, deltas, color='purple', lw=2)
    ax.axvline(K_g, color='black', linestyle='--', label='At-the-money')
    ax.set_title("Call Delta vs Stock Price")
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Delta")
    ax.legend()
    st.pyplot(fig)

