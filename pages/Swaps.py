import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ===========================
# Page Configuration
# ===========================
st.set_page_config(page_title="Swaps Dashboard", layout="wide")
st.title("📈 Swaps Dashboard")

# Custom CSS for polished UI (Matching your Bonds style)
st.markdown("""
<style>
div[data-baseweb="select"] > div {
    color: #1a73e8 !important;
    font-size: 18px !important;
    font-weight: bold;
}
.stMetric {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# Navigation
# ===========================
section = st.selectbox(
    "Select Subsection:",
    [
        "Foundations of Swaps",
        "Foreign Exchange (FX) Swaps",
        "Interest Rate Swaps (IRS)",
        "Credit Default Swaps (CDS)"
    ]
)

# =====================================================
# 1️⃣ FOUNDATIONS – THE ARCHITECTURE OF SWAPS
# =====================================================
if section == "Foundations of Swaps":
    st.header("📘 Foundations of Swap Contracts")
    
  
    st.markdown("""
A **swap** is a private agreement (bilateral) between two parties to exchange cash flows in the future based on a pre-defined rule. 

Unlike options, which give one party the *right* but not the obligation to act, swaps are **forward-based contracts**. This means **both parties have a binding obligation** to make payments to each other over the life of the contract.

### 🔹 Core Structural Elements
Every swap is defined by its "Legs." Think of a leg as one side of the exchange.
* **The Fixed Leg:** One party agrees to pay a set, unchangeable rate ($r$) on a specific notional amount ($L$) at regular intervals.
* **The Floating Leg:** The other party agrees to pay a rate that changes over time, usually linked to a market benchmark (like SOFR, LIBOR, or an FX rate).



Because of this structure, swaps are primarily used to transform a liability (like changing a variable-rate loan to a fixed-rate loan) rather than just speculating on price movements.
""")
        

    st.markdown("---")

    # =====================================================
    # THEORETICAL REPLICATION
    # =====================================================
    st.subheader("🛠️ The Building Block View (Replication)")
    st.markdown("""
    To price or hedge a swap, financial engineers view them as a combination of simpler instruments. 
    A swap can be mathematically decomposed in two primary ways:
    """)

    rep_col1, rep_col2 = st.columns(2)
    
    with rep_col1:
        st.markdown("#### 1. Portfolio of Forwards")
        st.write("""
        A swap is essentially a bundle of **Forward Rate Agreements (FRAs)** or Forward contracts. 
        Each payment date is a separate forward, but they all share the same "delivery price" (the swap rate).
        """)
        st.latex(r"V_{swap} = \sum_{i=1}^{n} Forward_i")
        
    with rep_col2:
        st.markdown("#### 2. Portfolio of Bonds")
        st.write("""
        A "Pay-Fixed" swap is economically identical to **Shorting a Fixed-Rate Bond** and using the proceeds to buy a **Floating Rate Note (FRN)**.
        """)
        st.latex(r"V_{swap} = V_{floating\_bond} - V_{fixed\_bond}")

    st.markdown("---")

    # =====================================================
    # NO-ARBITRAGE & MTM
    # =====================================================
    st.subheader("⚖️ No-Arbitrage Pricing & Mark-to-Market")
    
    st.markdown("""
    At inception ($t=0$), the swap rate is set such that the **Net Present Value (NPV)** is zero. 
    This is known as the **Fair Swap Rate**.
    """)
    
    st.latex(r"V_{Swap}(0) = \sum PV(Floating) - \sum PV(Fixed) = 0")
    
    st.warning("""
    **The Lifecycle of a Swap:** As soon as market interest rates or currency prices move, one leg becomes more valuable than the other. 
    The swap now has a **Mark-to-Market (MTM)** value. 
    If $V_{swap} > 0$, the swap is an **Asset**; if $V_{swap} < 0$, it is a **Liability**.
    """)

    st.markdown("---")

    # =====================================================
    # COMPARATIVE ADVANTAGE
    # =====================================================
    st.subheader("📈 Comparative Advantage & Quality Spread Differential (QSD)")
    
    st.markdown("""
    The classic theory for the existence of swaps is that different firms have relative 
    advantages in different funding markets.
    """)

    qsd_data = {
        "Market": ["Fixed Rate Market", "Floating Rate Market"],
        "Company A (AAA Rated)": ["4.00%", "SOFR + 0.10%"],
        "Company B (BBB Rated)": ["5.50%", "SOFR + 0.60%"],
        "Credit Spread": ["1.50%", "0.50%"]
    }
    
    df = pd.DataFrame(qsd_data).set_index("Market")

    st.table(df)

    st.markdown("""
    **Analysis:**
    - Company A has an absolute advantage in **both** markets (lower rates).
    - However, Company A's advantage is **larger** in the Fixed market (1.50% vs 0.50%).
    - **QSD** = $|1.50\% - 0.50\%| = 1.00\%$.
    
    By swapping, the companies can split this **1.00% surplus**, allowing both to borrow 
    at rates lower than they could achieve individually.
    """)

    st.caption("Theory Note: In modern markets, QSD has diminished due to increased market efficiency, but remains a foundational concept for understanding financial intermediation.")

# =====================================================
# 2️⃣ FX SWAPS – LIQUIDITY MANAGEMENT & MULTI-PERIOD PRICING
# =====================================================
elif section == "Foreign Exchange (FX) Swaps":
    st.header("💱 FX Swaps: Liquidity & Multi-Period Pricing")
    
    st.markdown("""
    An FX Swap is a simultaneous agreement to exchange currencies at two different dates. 
    It is the most traded instrument in the FX market because it allows institutions to **manage liquidity** without taking on "directional risk" (betting on which way the currency goes).
    """)

    # --- CLARIFIED SWAP STRUCTURES ---
    st.subheader("🔹 Understanding the 'Legs'")
    st.markdown("""
    Think of an FX Swap as a **temporary trade**. You give something away today, but you agree 
    exactly when and at what price you will get it back.
    """)
    
    col_struct1, col_struct2 = st.columns(2)
    with col_struct1:
        st.info("**Type A: Buy-Sell Swap**")
        st.markdown("""
        * **Near Leg (Today):** You **Buy** Foreign Currency at the Spot rate.
        * **Far Leg (Future):** You **Sell** it back at the Forward rate.
        * *Analogy:* You are "borrowing" Foreign Currency and using your Domestic cash as collateral.
        """)

    with col_struct2:
        st.info("**Type B: Sell-Buy Swap**")
        st.markdown("""
        * **Near Leg (Today):** You **Sell** Foreign Currency at the Spot rate.
        * **Far Leg (Future):** You **Buy** it back at the Forward rate.
        * *Analogy:* You are "lending" Foreign Currency to earn the interest differential.
        """)

    st.markdown("---")

    # --- DETAILED THEORY: THE UNIFORM RATE X ---
    st.subheader("⚖️ Deep Dive: The Uniform Swap Rate ($X$)")
    st.markdown("""
    When a company enters a **Multi-Period Swap** (e.g., exchanging BRL for USD every month for a year), 
    it is messy to use 12 different Forward rates. Instead, banks quote a single **Uniform Rate ($X$)**.
    
    ### Where does $X$ come from?
    As per the **No-Arbitrage Principle**, the Present Value (PV) of all cash flows in a swap must equal zero at inception. 
    If we are receiving a foreign amount ($C_f$) and paying a domestic amount ($C_f \cdot X$):
    """)
    
    st.latex(r"PV = \sum_{j=1}^{N} \left[ \text{PV of Receiving Foreign} - \text{PV of Paying Domestic} \right] = 0")
    st.latex(r"\sum_{j=1}^{N} \left[ (C_f \cdot F_j \cdot DF_j) - (C_f \cdot X \cdot DF_j) \right] = 0")

    st.markdown("""
    By cancelling out the constant notional $C_f$ and solving for $X$, we get:
    """)
    st.latex(r"X = \frac{\sum_{j=1}^{N} F_j \cdot DF_j}{\sum_{j=1}^{N} DF_j}")

    with st.expander("Variable Definitions"):
        st.markdown("""
        - **$X$**: The Uniform Swap Rate (The single exchange rate used for all periods).
        - **$F_j$**: The theoretical Forward Rate for period $j$ (calculated via CIP).
        - **$DF_j$**: The Domestic Discount Factor for period $j$ ($e^{-r_d \cdot T_j}$). 
        - **Why $DF_{domestic}$?** Because the valuation is performed from the perspective of the domestic investor (USD).
        """)

    st.markdown("---")

    # --- CALCULATION & INTERPRETATION ---
    st.subheader("🔢 Numerical Step-by-Step: The BRL/USD Swap")
    
    # Sidebar inputs remain the same
    st.sidebar.markdown("**Market Parameters:**")
    notional_f = st.sidebar.number_input("Foreign Notional (BRL)", value=100000)
    periods = st.sidebar.slider("Number of Monthly Payments", 1, 12, 6)
    s_curr = st.sidebar.number_input("Current Spot (USD/BRL)", value=0.1850, format="%.4f")
    r_usd = st.sidebar.number_input("USD Rate (Domestic) %", value=5.25) / 100
    r_brl = st.sidebar.number_input("BRL Rate (Foreign) %", value=12.00) / 100
        
    data = []
    sum_f_df = 0
    sum_df = 0
        
    for i in range(1, periods + 1):
        t = i / 12
        # CIP: F = S * exp((rd-rf)t)
        f_i = s_curr * np.exp((r_usd - r_brl) * t)
        df_i = np.exp(-r_usd * t)
        
        weighted_f = f_i * df_i
        sum_f_df += weighted_f
        sum_df += df_i
        
        data.append({
            "Month": i, 
            "Forward Rate (F)": round(f_i, 5), 
            "Disc. Factor (DF)": round(df_i, 4),
            "F × DF": round(weighted_f, 5)
        })
    
    x_rate = sum_f_df / sum_df
        
    col_calc1, col_calc2 = st.columns([2, 1])
    
    with col_calc1:
        st.table(pd.DataFrame(data).set_index("Month"))
    
    with col_calc2:
        st.markdown("**Calculation Breakdown:**")
        st.write(f"Numerator ($\sum F \cdot DF$): `{sum_f_df:.4f}`")
        st.write(f"Denominator ($\sum DF$): `{sum_df:.4f}`")
        st.markdown(f"### $X = {x_rate:.5f}$")
        
    st.markdown("---")
    
    # --- WHY IS THE FORWARD RATE DECREASING? ---
    st.subheader("❓ Why is the Forward Rate decreasing over time?")
    
    st.markdown(f"""
    In this example, the USD rate is **{r_usd*100:.2f}%** and the BRL rate is **{r_brl*100:.2f}%**.
    
    1.  **Interest Rate Differential:** The foreign currency (BRL) offers a much higher yield than the domestic currency (USD).
    2.  **The No-Arbitrage Rule:** If you hold BRL, you earn an extra { (r_brl - r_usd)*100 :.2f}% per year.
    3.  **The Adjustment:** To prevent investors from simply "farming" the high BRL interest rate, the **Forward Price must drop**. 
    
    This drop ensures that the gain you make from high BRL interest is exactly offset by the loss you take when you convert those BRL back into USD at a lower price in the future.
    """)

    st.info(f"The 'Forward Discount' on BRL is roughly {(r_brl - r_usd)*100:.2f}% per year.")


# =====================================================
# 3️⃣ INTEREST RATE SWAPS (IRS) – VALUATION & PAR RATES
# =====================================================
elif section == "Interest Rate Swaps (IRS)":
    st.header("🔄 IRS Valuation Framework")
    
    st.markdown("""
    An Interest Rate Swap (IRS) is a contract to exchange a **Fixed** interest rate for a **Floating** interest rate (e.g., SOFR). 
    Like an FX swap, it is a bilateral OTC agreement where only the **net** difference in interest is usually exchanged.
    """)

    # --- THE INTUITION OF LEGS ---
    st.subheader("🔹 Understanding the 'Legs'")
    st.markdown("""
    Think of an IRS as two separate bonds bundled into one contract. 
    One party "sells" a fixed-rate bond and "buys" a floating-rate bond (or vice versa).
    """)
    
    col_leg1, col_leg2 = st.columns(2)
    with col_leg1:
        st.info("**The Fixed Leg**")
        st.markdown("""
        * **Payment:** Pre-determined interest rate ($K$).
        * **Certainty:** Known at inception; does not change regardless of market moves.
        * **Analogy:** Like a fixed-rate mortgage payment.
        """)

    with col_leg2:
        st.info("**The Floating Leg**")
        st.markdown("""
        * **Payment:** Based on a market benchmark (e.g., SOFR + Spread).
        * **Certainty:** Unknown for future periods; resets periodically.
        * **Analogy:** Like an adjustable-rate loan.
        """)

    st.markdown("---")

    # --- THE PAR SWAP RATE DERIVATION ---
    st.subheader("⚖️ Deep Dive: The Par Swap Rate ($K$)")
    st.markdown("""
    Just like the Uniform Rate ($X$) in FX swaps, the **Par Swap Rate ($K$)** is the fixed rate that makes the 
    Present Value (PV) of the swap equal to **zero** at inception.
    
    ### The Mathematical Identity
    A fundamental rule in finance is that a **Floating Rate Note (FRN)**—which pays the market rate—is always worth **Par ($1.00$)** on its reset dates. We use this to solve for $K$:
    """)
    
    st.latex(r"PV_{Floating} = PV_{Fixed}")
    st.latex(r"1 = \sum_{i=1}^{n} (K \cdot \alpha_i \cdot DF_i) + DF_n")

    st.markdown("Solving for $K$, we find that the fair swap rate is the ratio of the 'unpaid principal' to the 'annuity' of discount factors:")
    st.latex(r"K = \frac{1 - DF_n}{\sum_{i=1}^{n} \alpha_i \cdot DF_i}")

    with st.expander("Variable Definitions"):
        st.markdown("""
        - **$K$**: The Par Swap Rate (The 'fair' rate fixed today).
        - **$DF_n$**: The Discount Factor for the final maturity date.
        - **alpha**: The accrual factor (e.g., $0.5$ for semi-annual, $1.0$ for annual).
        - **$DF_i$**: The Discount Factor for each payment date $i$.
        """)

    st.markdown("---")

    # --- NUMERICAL CALCULATION ---
    st.subheader("🔢 Numerical Step-by-Step: The Par Rate Calculation")
    
    st.sidebar.markdown("**IRS Market Parameters:**")
    notional = st.sidebar.number_input("Notional Amount ($)", value=10000000)
    years = st.sidebar.slider("Tenor (Years)", 1, 10, 5)
    mkt_yield = st.sidebar.number_input("Market Yield (Flat Curve %)", value=4.5) / 100

    # Building the discount curve
    data_irs = []
    sum_df_alpha = 0
    for i in range(1, years + 1):
        df_i = np.exp(-mkt_yield * i) # Continuous discounting
        alpha = 1.0 # Annual payments
        sum_df_alpha += (df_i * alpha)
        data_irs.append({
            "Year": i, 
            "Discount Factor (DF)": round(df_i, 5), 
            "Alpha": alpha,
            "Weighted DF (DF × α)": round(df_i * alpha, 5)
        })
    
    df_n = data_irs[-1]["Discount Factor (DF)"]
    par_k = (1 - df_n) / sum_df_alpha

    col_res1, col_res2 = st.columns([2, 1])
    with col_res1:
        st.table(pd.DataFrame(data_irs).set_index("Year"))
    
    with col_res2:
        st.markdown("**Calculation Breakdown:**")
        st.write(f"1 - Final DF ($1 - DF_n$): `{1 - df_n:.4f}`")
        st.write(f"Total Annuity ($\sum DF \cdot α$): `{sum_df_alpha:.4f}`")
        st.write(f"### Par Swap Rate: {par_k*100:.4f}%")
        
    st.info(f"""
    **Result Interpretation:**
    If you enter a {years}-year swap today, the 'fair' fixed rate is **{par_k*100:.2f}%**. 
    At this rate, the value of the fixed payments you make exactly equals the value of the floating payments 
    you expect to receive, making the swap's initial value **zero**.
    """)

# =====================================================
# 4️⃣ CREDIT DEFAULT SWAPS (CDS) – PROTECTION & HAZARD RATES
# =====================================================
elif section == "Credit Default Swaps (CDS)":
    st.header("🛡️ Credit Default Swaps: Credit Insurance & Systemic Risk")
    
    st.markdown("""
    A **Credit Default Swap (CDS)** is a financial derivative that allows an investor to "swap" their credit risk with another investor. 
    While it functions like insurance, it is a tradable swap contract. 
    
    **Historical Context:** CDS gained global notoriety as a central driver of the **2008 Financial Crisis**. 
    Institutions sold massive amounts of protection on Mortgage-Backed Securities (MBS) without holding enough capital 
    to cover the "Floating Leg" (payouts) when default rates spiked.
    """)

    # --- THE INTUITION OF LEGS ---
    st.subheader("🔹 Understanding the 'Legs'")
    st.markdown("""
    Unlike an IRS where interest is exchanged for interest, a CDS exchanges a certain payment for a contingent one.
    """)
    
    col_cds1, col_cds2 = st.columns(2)
    with col_cds1:
        st.info("**The Premium Leg (Fixed)**")
        st.markdown("""
        * **Payer:** Protection Buyer.
        * **Payment:** A fixed periodic fee called the **CDS Spread**.
        * **Duration:** Paid until the contract expires or a default occurs.
        """)

    with col_cds2:
        st.info("**The Protection Leg (Floating)**")
        st.markdown("""
        * **Payer:** Protection Seller.
        * **Payment:** Only triggered by a **Credit Event** (Bankruptcy, Failure to Pay, etc.)
        """)
    
    

    st.markdown("---")

    # --- THE MATHEMATICAL MODEL: HAZARD RATES ---
    st.subheader("⚖️ The Hazard Rate Model ($\lambda$)")
    st.markdown("""
    To price a CDS, we model the probability of default using a **Hazard Rate ($\lambda$)**. 
    The Hazard rate represents the "instantaneous" probability of default.
    """)
    
    st.latex(r"P(\text{Survival until time } t) = e^{-\lambda t}")
    st.latex(r"P(\text{Default by time } t) = 1 - e^{-\lambda t}")

    st.markdown("""
    ### The Pricing Identity
    At inception, the PV of the Premium Leg must equal the PV of the Protection Leg. 
    For a single period, the "rule of thumb" relationship is:
    """)
    st.latex(r"\text{CDS Spread} \approx \lambda \times (1 - R)")
    
    with st.expander("Variable Definitions"):
        st.markdown("""
        - **Spread**: The annual premium (expressed in basis points).
        - **$\lambda$ (Hazard Rate)**: The implied likelihood of default per year.
        - **$R$ (Recovery Rate)**: The percentage of the debt recovered after default (Standard is often 40%).
        - **$(1 - R)$**: The 'Loss Given Default' (LGD).
        """)

    st.markdown("---")

    # --- NUMERICAL CALCULATION ---
    st.subheader("🔢 Numerical Step-by-Step: Implied Default Probabilities")
    
    st.sidebar.markdown("**CDS Market Parameters:**")
    spread_bps = st.sidebar.slider("Market CDS Spread (bps)", 10, 2000, 200)
    recovery_rate = st.sidebar.slider("Assumed Recovery Rate (%)", 0, 100, 40) / 100
    tenor_cds = st.sidebar.number_input("Tenor (Years)", value=5)

    # Calculation
    # 1 bp = 0.0001
    annual_spread = spread_bps / 10000
    implied_hazard = annual_spread / (1 - recovery_rate)
    
    # Building a survival table
    data_cds = []
    for t in range(1, tenor_cds + 1):
        surv = np.exp(-implied_hazard * t)
        data_cds.append({
            "Year": t,
            "Survival Probability": f"{surv*100:.2f}%",
            "Cumulative Default Prob": f"{(1 - surv)*100:.2f}%"
        })

    col_calc_cds1, col_calc_cds2 = st.columns([2, 1])
    with col_calc_cds1:
        st.table(pd.DataFrame(data_cds).set_index("Year"))
    
    with col_calc_cds2:
        st.markdown("**Valuation Summary:**")
        st.write(f"Implied Hazard Rate ($\lambda$): `{implied_hazard:.4f}`")
        st.write(f"Loss Given Default ($1-R$): `{1-recovery_rate:.2f}`")
        st.write(f"### Annual Default Prob: {implied_hazard*100:.2f}%")

    st.info(f"""
    **Result Interpretation:**
    A CDS spread of **{spread_bps} bps** implies that the market sees a **{implied_hazard*100:.2f}%** chance of this entity defaulting every year. Over {tenor_cds} years, there is a 
    **{(1 - np.exp(-implied_hazard * tenor_cds))*100:.2f}%** cumulative chance you will have to pay out on the protection leg.
    """)
