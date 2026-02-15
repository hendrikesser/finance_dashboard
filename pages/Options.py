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
        "Black-Scholes Model and Implied Volatility",
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
    st.header("🌳 Binomial Options Pricing Model")
    
    st.markdown("""
    The Binomial Model is a discrete-time framework that assumes over a small interval $\Delta t$, the price moves **Up ($u$)** or **Down ($d$)**. 
    This specific implementation uses a **recombining tree**, where an 'Up' move followed by a 'Down' move returns to the original price ($u \cdot d = 1$).
    """)

    # --- Sidebar Parameters ---
    st.sidebar.header("Tree Parameters")
    S0 = st.sidebar.number_input("Initial Stock Price ($S_0$)", value=100.0)
    sigma_b = st.sidebar.slider("Volatility ($\sigma$)", 0.1, 0.8, 0.2)
    r_b = st.sidebar.slider("Risk-free rate ($r$)", 0.0, 0.2, 0.05)
    T_b = st.sidebar.slider("Time to Maturity ($T$ in years)", 0.1, 2.0, 1.0)
    K_b = st.sidebar.number_input("Strike Price ($K$)", value=100.0)

    # --- 1-PERIOD BINOMIAL TREE (European Call for Intro) ---
    st.subheader("1️⃣ The 1-Period Model")
    dt1 = T_b
    u1 = np.exp(sigma_b * np.sqrt(dt1))
    d1 = 1/u1
    q1 = (np.exp(r_b * dt1) - d1) / (u1 - d1)
    
    Su, Sd = S0 * u1, S0 * d1
    Cu, Cd = max(Su - K_b, 0), max(Sd - K_b, 0)
    C0 = np.exp(-r_b * dt1) * (q1 * Cu + (1 - q1) * Cd)

    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown(f"**Risk-Neutral Probability ($q$):** {q1:.4f}")
        st.latex(r"q = \frac{e^{r\Delta t} - d}{u - d}")
        st.write(f"This $q$ ensures the asset's expected return is $r$.")
        st.metric("1-Period Call Price", f"{C0:.2f}")
    with col2:
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.plot([0, 1], [S0, Su], 'bo-'); ax1.plot([0, 1], [S0, Sd], 'bo-')
        ax1.text(1.05, Su, f'S_u={Su:.1f}\nC_u={Cu:.2f}'); ax1.text(1.05, Sd, f'S_d={Sd:.1f}\nC_d={Cd:.2f}')
        ax1.axis('off')
        st.pyplot(fig1)

    st.markdown("---")

    # --- 2-PERIOD BINOMIAL TREE (American Put) ---
    st.subheader("2️⃣ The 2-Period American Put Model")
    st.write("Building on the 1-period logic, we now allow for early exercise at each node, as well as more steps, which is crucial for American options.")
    st.info("💡 **American Option Logic:** At each node, we check if $Payoff_{Exercise} > Value_{Hold}$.")
    
    dt2 = T_b / 2
    u2 = np.exp(sigma_b * np.sqrt(dt2))
    d2 = 1/u2
    q2 = (np.exp(r_b * dt2) - d2) / (u2 - d2)
    disc = np.exp(-r_b * dt2)

    # Node Calculations - Stock Prices
    S_u, S_d = S0 * u2, S0 * d2
    S_uu, S_ud, S_dd = S0*(u2**2), S0, S0*(d2**2) # S_ud is S0 because u*d=1 (Recombining)

    # Step 1: Payoffs at Expiration (t=2)
    P_uu, P_ud, P_dd = max(K_b - S_uu, 0), max(K_b - S_ud, 0), max(K_b - S_dd, 0)

    # Step 2: Backward Induction to t=1
    # Node Up (u)
    cont_u = disc * (q2 * P_uu + (1-q2) * P_ud)
    exer_u = max(K_b - S_u, 0)
    P_u = max(cont_u, exer_u)

    # Node Down (d)
    cont_d = disc * (q2 * P_ud + (1-q2) * P_dd)
    exer_d = max(K_b - S_d, 0)
    P_d = max(cont_d, exer_d)

    # Step 3: Backward Induction to t=0
    cont_0 = disc * (q2 * P_u + (1-q2) * P_d)
    exer_0 = max(K_b - S0, 0)
    P0_2 = max(cont_0, exer_0)

    # Visualizing the Tree
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    nodes = {'0': (0, S0), 'u': (1, S_u), 'd': (1, S_d), 'uu': (2, S_uu), 'ud': (2, S_ud), 'dd': (2, S_dd)}
    for start, end in [('0','u'), ('0','d'), ('u','uu'), ('u','ud'), ('d','ud'), ('d','dd')]:
        ax2.plot([nodes[start][0], nodes[end][0]], [nodes[start][1], nodes[end][1]], 'gray', linestyle='--')

    ax2.text(0, S0, f"S={S0}\n**P={P0_2:.2f}**", bbox=dict(facecolor='lightgrey'))
    ax2.text(1, S_u, f"S_u={S_u:.1f}\nP_u={P_u:.2f} {'(Ex)' if exer_u > cont_u else ''}")
    ax2.text(1, S_d, f"S_d={S_d:.1f}\nP_d={P_d:.2f} {'(Ex)' if exer_d > cont_d else ''}")
    ax2.text(2, S_uu, f"S_uu={S_uu:.1f}\nPayoff={P_uu:.2f}")
    ax2.text(2, S_ud, f"S_ud={S_ud:.1f}\nPayoff={P_ud:.2f}")
    ax2.text(2, S_dd, f"S_dd={S_dd:.1f}\nPayoff={P_dd:.2f}")
    ax2.set_title("2-Step Recombining Tree (American Put)")
    st.pyplot(fig2)

    # Calculation Breakdown
    with st.expander("🔍 See Detailed Step-by-Step Calculations"):
        st.write(f"**Parameters:** $\Delta t = {dt2:.2f}$, $u = {u2:.4f}$, $d = {d2:.4f}$, $q = {q2:.4f}$")
        st.markdown(f"""
        1. **At t=2 (Final Nodes):**
           - $P_{{uu}} = \max({K_b} - {S_uu:.2f}, 0) = {P_uu:.2f}$
           - $P_{{ud}} = \max({K_b} - {S_ud:.2f}, 0) = {P_ud:.2f}$
           - $P_{{dd}} = \max({K_b} - {S_dd:.2f}, 0) = {P_dd:.2f}$
        2. **At t=1 (Backward Induction):**
           - Node Down: Continuation value = $e^{{-r\Delta t}}({q2:.2f} \cdot {P_ud} + {1-q2:.2f} \cdot {P_dd}) = {cont_d:.2f}$. 
           - Exercise value = ${K_b} - {S_d:.2f} = {exer_d:.2f}$.
           - **$P_d = \max({cont_d:.2f}, {exer_d:.2f}) = {P_d:.2f}$**
        """)

    st.markdown("---")
    st.subheader("♾️ The Theoretical Limit: Black-Scholes")
    st.write("The theoretical limit of the binomial model is the Black-Scholes model. As the number of steps increases, the binomial price converges to the Black-Scholes price. With this, our delta t approaches zero, while at the same time the volatility per step reduces, such that the overall variance over the life of the option remains constant.")
    st.write("In the next section, we will explore the continuous-time Black-Scholes model, which provides a closed-form solution for European options under certain assumptions.")
    
# =======================================
# 3️⃣ BLACK-SCHOLES MODEL and Implied Volatility
# =======================================
elif section == "Black-Scholes Model and Implied Volatility":
    st.header("🧪 Black-Scholes-Merton (BSM) Model")

    # --- 1. Introduction & Theoretical Link ---
    st.subheader("1️⃣ From Binomial Trees to Continuous Time")
    st.write("""
    The Black-Scholes model is the continuous-time limit of the Binomial Tree. 
    Imagine a tree where the time step $\Delta t$ becomes infinitely small and the number of periods $N$ becomes infinitely large.
    """)

    st.info("⚠️ **Important:** While the Binomial Tree can price American options, the standard Black-Scholes formula applies **only to European options**.")

    with st.expander("📝 Short Derivation Sketch"):
        st.write("""
        1. **Price Movement:** In each step of a tree, $u = e^{\sigma\sqrt{dt}}$ and $d = e^{-\sigma\sqrt{dt}}$. 
        2. **Distribution:** According to the **Central Limit Theorem**, as $N \to \infty$, the sum of these discrete up/down moves converges to a Normal distribution.
        3. **Log-Normality:** Specifically, the log of the stock price at maturity follows:
        """)
        st.latex(r"\ln(S_T) \sim N\left(\ln(S_0) + (r - y - \frac{\sigma^2}{2})T, \sigma^2 T\right)")
        st.write("""
        4. **Pricing:** The option price is then the expected payoff under this distribution, discounted back at the risk-free rate: $e^{-rT} E[max(S_T - K, 0)]$.
        """)

    # --- NEW: BSM ASSUMPTIONS ---
    st.subheader("📂 Core Assumptions of the BSM Economy")
    st.markdown("""
    To arrive at a closed-form solution, Black, Scholes, and Merton assumed a "frictionless" market:
    * **Log-Normal Returns:** Stock price follows Geometric Brownian Motion; log returns are independent and normally distributed.
    * **Constant Parameters:** Volatility ($\sigma$), risk-free rate ($r$), and dividend yield ($y$) are known and stay constant over the option's life.
    * **No Arbitrage:** There are no riskless profit opportunities.
    * **Continuous Trading:** No transaction costs, no taxes, and securities are infinitely divisible (you can buy 0.0001 shares).
    * **No Early Exercise:** Strictly for European-style options.
    """)

    st.markdown("---")

    # --- 2. The Formula & Components ---
    st.subheader("2️⃣ The Black-Scholes Formula")
    
    st.write("The price of a European Call ($C$) and Put ($P$) is given by:")
    st.latex(r"C = S_0 e^{-yT} N(d_1) - K e^{-rT} N(d_2)")
    st.latex(r"P = K e^{-rT} N(-d_2) - S_0 e^{-yT} N(-d_1)")
    
    st.write("**How to calculate $d_1$ and $d_2$:**")
    st.latex(r"d_1 = \frac{\ln(S_0/K) + (r - y + \sigma^2/2)T}{\sigma\sqrt{T}} \quad , \quad d_2 = d_1 - \sigma\sqrt{T}")

    with st.expander("🔍 Explaining the Components"):
        st.markdown("""
        * **$N(x)$**: The cumulative standard normal distribution (probability that a variable is $\le x$).
        * **$N(d_2)$**: The **risk-neutral probability** that the option will expire in-the-money (for a call, $S_T > K$).
        * **$N(d_1)$**: The **option delta ($\Delta$)** for a non-dividend paying stock. It represents the hedge ratio—how many shares of the underlying stock are needed to hedge one call option.
        * **$K e^{-rT} N(d_2)$**: The present value of the strike price you expect to pay, multiplied by the probability of paying it.
        * **$S_0 e^{-yT} N(d_1)$**: The present value of the stock you expect to receive, weighted by its hedge ratio.
        * **$y$**: The continuous dividend yield.
        """)

    st.markdown("---")

    # --- 3. Numerical Example & Inputs ---
    # ... [Rest of your numerical example and sensitivity code remains the same] ...

    # --- 3. Numerical Example & Inputs ---
    st.subheader("3️⃣ Numerical Example")
    
    # Input columns
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        S_ex = st.number_input("Current Price ($S_0$)", value=4000.0)
        K_ex = st.number_input("Strike Price ($K$)", value=4200.0)
        T_ex = st.number_input("Maturity (Years)", value=0.5)
    with col_in2:
        r_ex = st.number_input("Risk-free Rate (e.g., 0.04)", value=0.04)
        y_ex = st.number_input("Dividend Yield (e.g., 0.017)", value=0.017)
        sigma_ex = st.number_input("Volatility (e.g., 0.20)", value=0.20)

    # Calculation logic
    d1_ex = (np.log(S_ex / K_ex) + (r_ex - y_ex + 0.5 * sigma_ex**2) * T_ex) / (sigma_ex * np.sqrt(T_ex))
    d2_ex = d1_ex - sigma_ex * np.sqrt(T_ex)
    
    call_ex = (S_ex * np.exp(-y_ex * T_ex) * norm.cdf(d1_ex) - 
               K_ex * np.exp(-r_ex * T_ex) * norm.cdf(d2_ex))
    
    put_ex = (K_ex * np.exp(-r_ex * T_ex) * norm.cdf(-d2_ex) - 
              S_ex * np.exp(-y_ex * T_ex) * norm.cdf(-d1_ex))

    # Display result
    st.write("### **Results**")
    res1, res2, res3 = st.columns(3)
    res1.metric("d1 parameter", f"{d1_ex:.4f}")
    res2.metric("d2 parameter", f"{d2_ex:.4f}")
    res3.metric("Call Price", f"${call_ex:.2f}")

    st.write(f"**Step-by-Step Calculation for this Example:**")
    st.latex(rf"d_1 = \frac{{\ln({S_ex}/{K_ex}) + ({r_ex} - {y_ex} + 0.5 \cdot {sigma_ex}^2) \cdot {T_ex}}}{{{sigma_ex} \cdot \sqrt{{{T_ex}}}}} = {d1_ex:.4f}")
    st.latex(rf"d_2 = {d1_ex:.4f} - {sigma_ex} \cdot \sqrt{{{T_ex}}} = {d2_ex:.4f}")
    st.write(f"Using $N(d_1) = {norm.cdf(d1_ex):.4f}$ and $N(d_2) = {norm.cdf(d2_ex):.4f}$ leads to a Call price of **${call_ex:.2f}**.")

    # --- 4. Volatility Sensitivity (Vega) ---
    st.markdown("---")
    st.subheader("📊 Price Sensitivity to Volatility")

    col_vis1, col_vis2 = st.columns([1, 1.5])

    with col_vis1:
        st.write("**Why does Volatility drive up prices?**")
        st.markdown("""
        An option has a **convex payoff** structure: 
        * Your losses are capped at the premium paid (the bottom).
        * Your potential gains are theoretically unlimited (the top).
        
        Higher volatility increases the "dispersion" of potential stock prices at maturity. Since you don't care how far the stock goes *below* the strike, but you benefit significantly the further it goes *above* the strike, more volatility makes the option more valuable.
        """)
        st.info("💡 In finance, this sensitivity is known as **Vega**.")

    with col_vis2:
        vols = np.linspace(0.05, 0.8, 50)
        
        # Recalculating prices for the graph based on the user's example inputs
        prices = []
        for v in vols:
            d1_v = (np.log(S_ex / K_ex) + (r_ex - y_ex + 0.5 * v**2) * T_ex) / (v * np.sqrt(T_ex))
            d2_v = d1_v - v * np.sqrt(T_ex)
            p = (S_ex * np.exp(-y_ex * T_ex) * norm.cdf(d1_v) - 
                 K_ex * np.exp(-r_ex * T_ex) * norm.cdf(d2_v))
            prices.append(p)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(vols * 100, prices, color='#1a73e8', lw=3)
        ax.set_xlabel("Annual Volatility (%)", fontsize=12)
        ax.set_ylabel("Call Price ($)", fontsize=12)
        ax.set_title("Vega: Option Price vs. Volatility", fontsize=14)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    st.markdown("---")

    st.header("📉 Implied Volatility (IV) & The VIX")
    
    st.markdown("""
    **Implied Volatility** is the "volatility parameter" $\sigma$ that, when plugged into the Black-Scholes formula, 
    makes the theoretical price equal to the **market price**. 
    
    Instead of using historical data to predict the future, we look at market prices to see what investors **expect** volatility to be over the life of the option.
    """)

    # --- IV Calculation Concept ---
    st.subheader("1️⃣ How is Implied Volatility calculated?")
    st.write("Since we cannot rearrange the Black-Scholes formula to solve for $\sigma$ directly, we use numerical methods like **Newton-Raphson** or **Goal Seek** in excel.")
    
    st.write("""
        **IV vs. Realized Volatility:**
        * **IV:** Forward-looking. Reflects the cost of insurance (the "Fear Gauge").
        * **Realized Vol:** Backward-looking. Measures actual past price swings.
        * *Note:* On average, IV > Realized Vol because investors pay a **Risk Premium** for protection.
        """)

    st.markdown("---")

    # --- The Volatility Smile ---
    st.subheader("2️⃣ The Volatility Smile & Skew")
    st.write("""
    If Black-Scholes were perfect, IV would be constant across all strike prices. In reality, we see a **Smile** (or Skew).
    This proves that the assumption of **Log-Normality** does not hold in the real market.
    """)

    col_sm1, col_sm2 = st.columns([1, 1.5])
    with col_sm1:
        st.markdown("""
        **Why the Smile exists:**
        * **Crash Phobia:** Investors associate a higher likelihood of market crashes than a normal distribution predicts.
        * **Expensive Puts:** Out-of-the-Money (OTM) puts are "excessively" expensive because they act as insurance against crashes.
        * **Negative Skew:** The market distribution has a much heavier "left tail" than Black-Scholes assumes.
        """)
    with col_sm2:
        # Generate a sample Volatility Smile/Skew plot
        strikes = np.linspace(80, 120, 20)
        # Typical equity skew: higher IV for lower strikes
        iv_skew = 0.25 - 0.002 * (strikes - 100) + 0.0001 * (strikes - 100)**2
        fig_sm, ax_sm = plt.subplots(figsize=(8, 5))
        ax_sm.plot(strikes, iv_skew, marker='o', color='#d93025', label="Market IV")
        ax_sm.axhline(y=0.20, color='gray', linestyle='--', label="BS Constant Vol")
        ax_sm.set_title("Equity Volatility Skew (S&P 500)")
        ax_sm.set_xlabel("Strike Price")
        ax_sm.set_ylabel("Implied Volatility")
        ax_sm.legend()
        ax_sm.grid(alpha=0.3)
        st.pyplot(fig_sm)

    st.markdown("---")

 # =======================================
# 5️⃣ GREEKS & RISK MANAGEMENT
# =======================================
elif section == "Greeks & Risk Management":
    st.header("🛡️ The Greeks & Delta Hedging")

    # --- 1. Theoretical Definitions ---
    st.subheader("1️⃣ What are the Greeks?")
    st.write("""
    The "Greeks" are partial derivatives of the Black-Scholes formula. They tell us exactly how 
    much the option price will change if one input (like stock price or volatility) moves.
    """)

    with st.expander("📚 Theoretical Explanation of Each Greek"):
        st.markdown("""
* **Delta ($\Delta$):** The sensitivity of the option price to a change in the underlying stock price ($\partial C / \partial S$). It is also the **hedge ratio** used to create a risk-less portfolio.

* **Gamma ($\Gamma$):** The sensitivity of Delta to a change in the stock price ($\partial^2 C / \partial S^2$). It measures the 'acceleration' of price moves and tells you how quickly you need to rebalance your hedge.

* **Vega ($\nu$):** The sensitivity to volatility ($\partial C / \partial \sigma$). It measures the "fear" premium in the option.

* **Theta ($\Theta$):** The sensitivity to the passage of time ($\partial C / \partial t$). Also known as 'time decay,' it represents the erosion of the option's value as it approaches expiration.

* **Rho ($\rho$):** The sensitivity to the risk-free interest rate ($\partial C / \partial r$).
""")

    st.markdown("---")

    # --- 2. Call vs. Put Greeks ---
    st.subheader("2️⃣ Connection Between Call and Put Greeks")
    st.write("""
    Through **Put-Call Parity**, we can derive a direct relationship between the Greeks of calls and puts. 
    Some Greeks are identical, while others are shifted.
    """)

    

    col_gp1, col_gp2 = st.columns(2)
    with col_gp1:
        st.markdown("**When they are the SAME:**")
        st.markdown("""
**Gamma ($\Gamma$):** The curvature is identical for both.

**Vega ($\nu$):** Volatility affects the 'optionality' equally for both.
        """)
        st.latex(r"\Gamma_{Call} = \Gamma_{Put}, \quad \nu_{Call} = \nu_{Put}")

    with col_gp2:
        st.markdown("**When they DIFFER:**")
        st.write("""
        * **Delta ($\Delta$):** Call Delta is positive ($e^{-yT}N(d_1)$), Put Delta is negative ($e^{-yT}(N(d_1)-1)$).
        * **Theta ($\Theta$):** Usually different because the 'cost of carry' ($r$) affects the strike payment differently.
        """)
        st.latex(r"\Delta_{Call} - \Delta_{Put} = e^{-yT}")

    st.markdown("---")

    # --- 3. Greek Calculations ---
    st.subheader("3️⃣ Calculate Your Greeks")
    
    # Inputs
    c1, c2, c3 = st.columns(3)
    with c1:
        S_g = st.number_input("Current Stock Price", value=100.0)
        K_g = st.number_input("Strike Price", value=100.0)
    with c2:
        T_g = st.number_input("Time to Maturity (Years)", value=0.5)
        sigma_g = st.number_input("Volatility (0.2 = 20%)", value=0.2)
    with c3:
        r_g = st.number_input("Risk-free Rate", value=0.04)
        y_g = st.number_input("Dividend Yield", value=0.02)

    # Logic
    d1 = (np.log(S_g / K_g) + (r_g - y_g + 0.5 * sigma_g**2) * T_g) / (sigma_g * np.sqrt(T_g))
    d2 = d1 - sigma_g * np.sqrt(T_g)
    pdf_d1 = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)

    delta_c = np.exp(-y_g * T_g) * norm.cdf(d1)
    delta_p = delta_c - np.exp(-y_g * T_g)
    gamma = (pdf_d1 * np.exp(-y_g * T_g)) / (S_g * sigma_g * np.sqrt(T_g))
    vega = S_g * np.exp(-y_g * T_g) * pdf_d1 * np.sqrt(T_g)

    res_c1, res_c2, res_c3, res_c4 = st.columns(4)
    res_c1.metric("Delta (Call)", f"{delta_c:.3f}")
    res_c2.metric("Delta (Put)", f"{delta_p:.3f}")
    res_c3.metric("Gamma", f"{gamma:.4f}")
    res_c4.metric("Vega", f"{vega:.3f}")

    st.markdown("---")

   # --- 4. Delta Hedging Simulator ---
    st.subheader("4️⃣ 🕹️ Portfolio Delta Hedging Simulator")
    
    # User Portfolio Setup
    st.markdown("##### **Set Your Portfolio (Positive = Buy, Negative = Sell/Write)**")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        n_calls = st.number_input("Quantity of Calls", value=10, step=1)
    with col_p2:
        n_puts = st.number_input("Quantity of Puts", value=0, step=1)

    if st.button("🚀 Run 10-Day Stock Simulation"):
        days = 10
        dt = 1/252
        prices = []
        portfolio_deltas = []
        
        # 1. DAY 1 CALCULATIONS
        current_s = S_g
        current_t = T_g
        
        d1_d1 = (np.log(current_s / K_g) + (r_g - y_g + 0.5 * sigma_g**2) * current_t) / (sigma_g * np.sqrt(current_t))
        delta_c1 = np.exp(-y_g * current_t) * norm.cdf(d1_d1)
        delta_p1 = delta_c1 - np.exp(-y_g * current_t)
        total_delta_d1 = (n_calls * delta_c1) + (n_puts * delta_p1)
        
        # 2. DAY 2 CALCULATIONS (Simulate one step)
        np.random.seed(None) 
        z_jump = np.random.normal()
        next_s = current_s * np.exp((r_g - y_g - 0.5 * sigma_g**2) * dt + sigma_g * np.sqrt(dt) * z_jump)
        next_t = current_t - dt
        
        d1_d2 = (np.log(next_s / K_g) + (r_g - y_g + 0.5 * sigma_g**2) * next_t) / (sigma_g * np.sqrt(next_t))
        delta_c2 = np.exp(-y_g * next_t) * norm.cdf(d1_d2)
        delta_p2 = delta_c2 - np.exp(-y_g * next_t)
        total_delta_d2 = (n_calls * delta_c2) + (n_puts * delta_p2)

        # --- EXPLAINER SECTION ---
        st.markdown("### 🧮 Understanding the Rebalancing Math")
        exp1, exp2 = st.columns(2)
        
        with exp1:
            with st.expander("📝 Day 1: The Initial Hedge"):
                st.write(f"**Stock Price ($S_1$):** ${current_s:.2f}")
                st.write(f"**Individual Deltas:**")
                st.latex(rf"\Delta_C = {delta_c1:.4f}, \quad \Delta_P = {delta_p1:.4f}")
                st.write(f"**Total Portfolio Delta:**")
                st.latex(rf"({n_calls} \times {delta_c1:.4f}) + ({n_puts} \times {delta_p1:.4f}) = {total_delta_d1:.3f}")
                st.success(f"**Initial Action:** You must hold **{-total_delta_d1:.2f}** shares to be Delta Neutral.")

        with exp2:
            with st.expander("🔄 Day 2: The Rebalance"):
                st.write(f"**Stock Price ($S_2$):** ${next_s:.2f}")
                st.write(f"**Updated Portfolio Delta:**")
                st.latex(rf"\Delta_{{Port}} = {total_delta_d2:.3f}")
                
                # The "How many to buy/sell" math
                current_hedge = -total_delta_d1
                target_hedge = -total_delta_d2
                diff = target_hedge - current_hedge
                
                st.write(f"**Rebalancing Math:**")
                st.latex(rf"{target_hedge:.2f} \text{{ (New target)}} - {current_hedge:.2f} \text{{ (Current shares)}} = {diff:+.2f}")
                st.warning(f"**Adjustment:** {'BUY' if diff > 0 else 'SELL'} {abs(diff):.2f} shares today.")

        # --- 3. FULL SIMULATION LOOP (Display Table) ---
        sim_prices = [current_s]
        sim_deltas = [total_delta_d1]
        
        # Already have Day 2 from above, now do the rest
        curr_s_loop = next_s
        curr_t_loop = next_t
        for i in range(1, days):
            sim_prices.append(curr_s_loop)
            d1_loop = (np.log(curr_s_loop / K_g) + (r_g - y_g + 0.5 * sigma_g**2) * curr_t_loop) / (curr_s_loop * np.sqrt(curr_t_loop))
            # (Calculation logic simplified for speed in loop)
            dc = np.exp(-y_g * curr_t_loop) * norm.cdf((np.log(curr_s_loop/K_g)+(r_g-y_g+0.5*sigma_g**2)*curr_t_loop)/(sigma_g*np.sqrt(curr_t_loop)))
            dp = dc - np.exp(-y_g * curr_t_loop)
            sim_deltas.append((n_calls * dc) + (n_puts * dp))
            
            curr_s_loop *= np.exp((r_g - y_g - 0.5 * sigma_g**2) * dt + sigma_g * np.sqrt(dt) * np.random.normal())
            curr_t_loop -= dt

        st.write("### **10-Day Rebalancing Log**")
        log_data = []
        for i in range(days):
            target_h = -sim_deltas[i]
            prev_h = 0 if i == 0 else -sim_deltas[i-1]
            adj = target_h - prev_h
            log_data.append({
                "Day": i + 1,
                "Stock Price": f"${sim_prices[i]:.2f}",
                "Port. Delta": f"{sim_deltas[i]:.3f}",
                "Current Shares": f"{prev_h:.2f}",
                "Required Shares": f"{target_h:.2f}",
                "Action": f"{'BUY' if adj > 0 else 'SELL'} {abs(adj):.2f}"
            })
        st.table(log_data)