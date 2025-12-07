import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ===========================
# Page Title & Theme
# ===========================
st.set_page_config(page_title="Futures Dashboard", layout="wide")
st.title("📉 Futures Dashboard")


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
        "Basics & Payoffs",
        "Stock and Commodity Futures",
        "FX & Interest Rate Futures",
        "Index & VIX Futures"
    ])
# =======================================
# SHARED FUNCTIONS
# =======================================

def futures_payoff_long(F0, K):
    """Payoff of long futures: + (F_T – F_0)"""
    return F0 - K

def futures_payoff_short(F0, K):
    """Payoff of short futures: + (K – F_T)"""
    return K - F0

def get_future_price_spot_costcarry(spot, r, storage, convenience, T):
    """
    Cost-of-carry model:
    F0 = S0 * exp( (r + storage - convenience) * T )
    """
    return spot * np.exp((r + storage - convenience) * T)

# =======================
# Sidebar Inputs
# =======================
if section == "Basics & Payoffs":
    st.sidebar.header("Basics & Payoffs Inputs")

    # Payoff Inputs
    st.sidebar.subheader("Payoff Parameters")
    F0 = st.sidebar.slider(
        "Initial Futures Price F₀", 
        min_value=1, max_value=100, value=50)
    payoff_qty = st.sidebar.slider("Quantity (Contract Size)", 1, 100, 10)
    FT_user = st.sidebar.slider(
        "Settlement Price Fᵀ",
        min_value=1, max_value=100, value=60)

    # Number of days for MtM simulation
    st.sidebar.markdown("---")
    st.sidebar.subheader("Mark-to-Market Parameters")
    days = st.sidebar.slider("Number of Days to Simulate", min_value=1, max_value=30, value=10)

    # Generate New Prices Button
    if "Ft_daily" not in st.session_state:
        # Initialize with F0 repeated
        st.session_state.Ft_daily = np.array([F0] * days)

    if st.sidebar.button("Generate New Prices", key="payoff_prices"):
        np.random.seed()
        daily_returns = np.random.normal(loc=0.0, scale=0.02, size=days)
        Ft = [F0]
        for r in daily_returns:
            Ft.append(Ft[-1] * (1 + r))
        st.session_state.Ft_daily = np.array(Ft[1:])
     # exclude initial F0

    # Use the stored daily prices
    Ft_daily = st.session_state.Ft_daily


elif section == "Stock and Commodity Futures":
    st.sidebar.header("Stock and Commodity Futures Inputs")
    S0 = st.sidebar.number_input("Spot Price S₀", 1, 5000, 100)
    r = st.sidebar.slider("Risk-free rate (%)", 0.0, 20.0, 3.0) / 100
    div_yield = st.sidebar.slider("Dividend/Convenience Yield (%)", 0.0, 10.0, 2.0) / 100
    storage = st.sidebar.slider("Storage Cost (%)", 0.0, 10.0, 1.0) / 100
    T = st.sidebar.slider("Maturity (years)", 0.1, 10.0, 1.0)

    F0_calc = get_future_price_spot_costcarry(S0, r, storage, div_yield, T)
 
   
elif section == "FX & Interest Rate Futures":
    st.sidebar.header("FX & Interest Rate Futures Inputs")

elif section == "Index & VIX Futures":
    st.sidebar.header("Index & VIX Futures Inputs")


## =======================================
# 1️⃣ BASICS & PAYOFFS SECTION
# =======================================
if section == "Basics & Payoffs":
    # -------------------------
    # THEORY SECTION
    # -------------------------
    st.header("📘 Futures Basics")

    st.write("""
    A **futures contract** is a standardized agreement between two parties to **buy or sell an asset at a predetermined price (F₀)** 
    on a specific future date.  It works similarly to a forward contract, but is traded on exchanges and uses additional mechanisms that make it safer and more liquid.
    """)

    st.markdown("---")

    # -----------------------------------------
    # Key Characteristics
    # -----------------------------------------
    st.subheader("🔹 Key Characteristics")

    st.write("""
    - **Standardized**: The exchange (CME, Eurex, ICE) sets contract size, quality, and delivery dates.     → Enables high liquidity and efficient trading.

    - **Two Parties**: A **long position** agrees to buy the asset, while a **short position** agrees to sell it at maturity.

    - **Mark-to-market**: Gains and losses are settled **daily** as futures prices move.    → This is the major difference from forwards.

    - **Zero initial price**: The contract always starts with **value = 0**. → The price of the contract (value) changes over time, but at initiation, no money changes hands.

    - **Low credit risk**: A central clearinghouse guarantees performance, reducing counterparty risk. The clearinghouse acts as a counterparty for both sides of the trade.

    - Cash vs physical settlement: Most futures are **cash-settled** (e.g., index futures), while some require **physical delivery** of the underlying asset (e.g., commodity futures).

    - **Highly liquid**: Widely used for **hedging**, **speculation**, and **arbitrage**.
    """)

    # -----------------------------------------
    # Futures Price vs Futures Value
    # -----------------------------------------
    st.subheader("🔹 Price vs. Value: The Key Distinction")

    st.write("""
    A futures position has **two moving components**:
    1. **Futures Price \(F_t\)** : The market-quoted price that moves with supply/demand and spot price changes (with the same maturity date)
    2. **Contract Value (your P/L)** :  Even if you locked in \(F_0\), your **position value changes daily**:
    - If **Fₜ > F₀** → long gains, short loses  
    - If **Fₜ < F₀** → short gains, long loses  

    This daily gain/loss is settled via **mark-to-market**.
    """)

    # -----------------------------------------
    # Forward Comparison Table
    # -----------------------------------------
    st.subheader("🔹 Futures vs. Forwards: Quick Comparison")

    d1, d2 = st.columns(2)
    with d1:
        st.write("""
        | Feature | **Futures** | **Forwards** |
        |--------|-------------|--------------|
        | Trading | Exchange | Over the Counter (OTC) |
        | Standardization | High | Low (custom contracts) |
        | Counterparty Risk | Very low (clearinghouse) | High |
        | Mark-to-Market | Yes (daily) | No |
        | Value at Initiation | Always **0** | Usually 0, sometimes not |
        | Liquidity | Very high | Low |
        | Typical Use | Hedging, speculation, arbitrage | Custom hedging needs |
        """)

    with d2:
        st.write("**Note:** In the following sections we will be focusing on **futures contracts** specifically. We will be ignoring forwards for the rest of this dashboard. Additionally, the interest on futures margin accounts is assumed to be zero for simplicity (if not stated otherwise).")


    st.markdown("---")

    # -------------------------
    # PAYOFF STRUCTURE
    # -------------------------
    st.subheader("🔹 Payoff Structure")

    st.write("""
        Like shortly discussed in the basics sections of Futures, the payoff is determined by the current future price (with the same maturity date). This gain depends on whether you take a **long** (buy) or **short** (sell) position.
    """)

    st.latex(r"\text{Long Payoff: } \pi_L = F_T - F_0") 
    st.write("Intuition: If the current Future price/settlement price \(F_T\) is higher than your locked-in price \(F_0\), you profit. If it's lower, you incur a loss.")
    st.latex(r"\text{Short Payoff: } \pi_S = F_0 - F_T")
    st.write("Intuition: If the current Future price/settlement price \(F_T\) is lower than your locked-in price \(F_0\), you profit. If it's higher, you incur a loss.\n\n")
    st.markdown("")

    # -------------------------
    # PAYOFF DIAGRAM
    # -------------------------
    st.subheader("📊 Combined Payoff Diagram (Long & Short Futures)")

    # -------------------------
    # Dynamic x-range based on F0 and user input
    padding_factor = 0.2  # 20% extra on each side
    F_min = min(F0, FT_user) * (1 - padding_factor)
    F_max = max(F0, FT_user) * (1 + padding_factor)
    F_T = np.linspace(F_min, F_max, 400)

    # -------------------------
    # PAYOFF QUANTITY
    long_payoff = (F_T - F0) * payoff_qty
    short_payoff = (F0 - F_T) * payoff_qty

    user_long_pl = (FT_user - F0) * payoff_qty
    user_short_pl = (F0 - FT_user) * payoff_qty

    # -------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Payoff Summary")
        st.write("""
        This diagram illustrates how **long** and **short** futures positions behave when the 
        settlement price changes. The quantity multiplier magnifies gains and losses.
        """)
        st.write("#### Inputs:")
        st.write(f"- **Futures Price F₀:** {F0}")
        st.write(f"- **Quantity:** {payoff_qty}")
        st.write(f"- **Settlement Price Fᵀ:** {FT_user}")

        h1, h2 = st.columns(2)
        with h1:
            st.write("#### Long Position Payoff:")
            st.latex(r"\pi_L = (F_T - F_0)\cdot Q")
            st.write(f"Payoff = ({FT_user} - {F0}) × {payoff_qty} = {user_long_pl:.2f}")
        with h2:
            st.write("#### Short Position Payoff:")
            st.latex(r"\pi_S = (F_0 - F_T)\cdot Q")
            st.write(f"Payoff = ({F0} - {FT_user}) × {payoff_qty} = {user_short_pl:.2f}")
    st.write("The Long Position payoff is equal to the opposite of the Short Position payoff.")
        

    with col2:
        st.subheader("Payoff Diagram")
        fig, ax = plt.subplots(figsize=(6, 4))

        # Colors
        light_blue = "#82caff"
        light_orange = "#ffb77a"
        dark_blue = "#004c99"
        dark_orange = "#cc5a00"

        # Plot lines
        ax.plot(F_T, long_payoff, linewidth=2.5, color=light_blue, label="Long Futures")
        ax.plot(F_T, short_payoff, linewidth=2.5, color=light_orange, label="Short Futures")

        # Breakeven & zero lines
        ax.axvline(F0, linestyle="--", color="gray", linewidth=1)
        ax.axhline(0, color="black", linewidth=1)

        # User payoff points
        ax.scatter(FT_user, user_long_pl, s=120, color=dark_blue, edgecolor="black", zorder=5)
        ax.scatter(FT_user, user_short_pl, s=120, color=dark_orange, edgecolor="black", zorder=5)

        # Axes & grid
        ax.set_xlabel("Settlement Price Fᵀ")
        ax.set_ylabel("Profit / Loss")
        ax.grid(alpha=0.25)

        # Y-limits: include user points + payoff lines
        all_y = np.concatenate([long_payoff, short_payoff, [user_long_pl, user_short_pl]])
        y_min, y_max = all_y.min(), all_y.max()
        padding = max((y_max - y_min) * 0.1, 1.0)
        ax.set_ylim(y_min - padding, y_max + padding)

        # X-limits: same as dynamic F_T
        ax.set_xlim(F_min, F_max)

        # Legend
        ax.legend(fontsize=8)
        st.pyplot(fig)



    st.markdown("---")
    # ========================================
    # Daily Settlement & Mark-to-Market
    # ========================================

    # -------------------------
    # MARK-TO-MARKET SECTION
    # -------------------------

    st.header("📈 Daily Settlement & Mark-to-Market (MtM)")
    st.write("Futures contracts are **marked-to-market daily**, meaning gains and losses are settled every day.")

    # -------------------------
    # Include initial F0
    # -------------------------
    Ft_all = np.insert(Ft_daily, 0, F0)
    daily_pl = np.diff(Ft_all) * payoff_qty
    cumulative_pl = np.cumsum(daily_pl)

    # Time index
    days_index = [f"Day {i}" for i in range(len(Ft_all))]

    # -------------------------
    # Create DataFrame with quantity column
    # -------------------------
    Ft_all_float = Ft_all.astype(float)

    # Previous day prices (shifted by 1)
    prev_day_price = np.insert(Ft_all_float[:-1], 0, np.nan)  # Day 0 has no previous day

    df_mtm = pd.DataFrame({
        "Futures Price Fₜ": Ft_all_float.round(2),
        "Previous Day Price Fₜ₋₁": prev_day_price.round(2),
        "Quantity Q": [payoff_qty] * len(Ft_all_float),
        "Daily P&L": np.insert(daily_pl, 0, 0).round(2),
        "Cumulative P&L": np.insert(cumulative_pl, 0, 0).round(2)
    }, index=days_index)




    s1, s2 = st.columns(2)
    with s1:
        st.subheader("🔹 Daily P&L Table (with F₀ and Quantity)")
        st.dataframe(df_mtm, use_container_width=True)
        st.write(f"**Total P&L after {days} days:** {cumulative_pl[-1]:.2f}")
    with s2:
        st.markdown("")
        st.markdown("")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(len(Ft_all)), np.insert(cumulative_pl, 0, 0), marker='o', color="#004c99", linewidth=2.5, label="Long Position")
        ax.set_xlabel("Day")
        ax.set_ylabel("Cumulative P/L")
        ax.set_title("Cumulative Daily P&L (Mark-to-Market)")
        ax.grid(alpha=0.25)
        ax.legend()
        st.pyplot(fig)


    st.markdown("---")



    # ========================================
    # HEDGING SECTION WITH NUMERICAL EXAMPLES
    # ========================================
    st.markdown("## 🔄 Long vs Short Futures – Full Hedge Comparison (Including Underlying Transaction)")

    st.write("""
    Futures hedging only makes sense when you also consider the **underlying transaction**:

    It is used to **lock in prices** for future purchases or sales of the underlying asset. In both cases, the futures payoff offsets adverse price movements in the underlying market.
    The seller will typically enter a **short futures position** to hedge against price declines, while the buyer will take a **long futures position** to hedge against price increases.

    Below we compare:
    1. **Unhedged outcome** (only the spot transaction)
    2. **Futures payoff**
    3. **Final hedged outcome (spot ± futures payoff)**  
    """)

    col1, col2 = st.columns(2)

    # ----------------------------------------------------
    # Long Hedge (Buyer of underlying)
    # ----------------------------------------------------
    with col1:
        st.subheader("📘 Long Hedge (Buyer Protecting Against Rising Prices)")

        # Futures payoff
        long_fut_payoff = payoff_qty * (FT_user - F0)

        # Underlying purchase at settlement
        underlying_cost_unhedged = payoff_qty * FT_user
        underlying_cost_hedged = underlying_cost_unhedged - long_fut_payoff

        st.markdown("### 🔍 Intuition")
        st.write("""
        A **long hedge** is used when you know you will **buy the underlying in the future**  
        and you fear that prices may **increase**.

        The futures payoff adjusts your final cost so that the **effective price stays near F₀**. Ensuring budget certainty.
        """)
      
        st.markdown("### 🧮 Step-by-Step Calculation")

        st.write(f"""
        **1️⃣ Unhedged Cost**  
        You buy the underlying at the future spot price:  
        \n
        - Cost = Q × Fᵀ  
        - = {payoff_qty} × {FT_user}  
        - = **{underlying_cost_unhedged:.2f}**
        """)

        st.write(f"""
        **2️⃣ Futures Payoff**  
        Long futures payoff = Q × (Fᵀ − F₀)  
        = {payoff_qty} × ({FT_user} − {F0})  
        = **{long_fut_payoff:.2f}**
        """)

        st.write(f"""
        **3️⃣ Total Hedged Cost**  
        Effective purchase cost = Underlying - Futures payoff  
        = {underlying_cost_unhedged:.2f} - {long_fut_payoff:.2f}  
        = **{underlying_cost_hedged:.2f}**
        """)
        st.write("Move the slider to see how different settlement prices affect outcomes (they don't change the effective costs!)")
        st.write("The futures payoff offsets any gain/loss from buying the underlying.")
        


    # ----------------------------------------------------
    # Short Hedge (Seller of underlying)
    # ----------------------------------------------------
    with col2:
        st.subheader("📕 Short Hedge (Seller Protecting Against Falling Prices)")

        # Futures payoff
        short_fut_payoff = payoff_qty * (F0 - FT_user)

        # Underlying revenue
        revenue_unhedged = payoff_qty * FT_user
        revenue_hedged = revenue_unhedged + short_fut_payoff

        st.markdown("### 🔍 Intuition")
        st.write("""
        A **short hedge** is used when you will **sell the underlying in the future**  
        and you fear that prices may **drop**.

        The futures payoff offsets the lower selling price, keeping your effective  
        selling price close to **F₀**.
        """)

        st.markdown("### 🧮 Step-by-Step Calculation")

        st.write(f"""
        **1️⃣ Unhedged Revenue**  
        You sell at the future spot price:  
        - Revenue = Q × Fᵀ  
        - = {payoff_qty} × {FT_user}  
        - = **{revenue_unhedged:.2f}**
        """)

        st.write(f"""
        **2️⃣ Hedged: Futures Payoff**  
        Short payoff = Q × (F₀ − Fᵀ)  
        = {payoff_qty} × ({F0} − {FT_user})  
        = **{short_fut_payoff:.2f}**
        """)

        st.write(f"""
        **3️⃣ Total Hedged Revenue**  
        Hedged Revenue = Underlying Sale + Futures payoff  
        = {revenue_unhedged:.2f} + {short_fut_payoff:.2f}  
        = **{revenue_hedged:.2f}**
        """)
        st.write("Move the slider to see how different settlement prices affect outcomes (they don't change the effective revenues!)")
        st.write("The futures payoff offsets any gain/loss from selling the underlying.")

    # ===========================
    # SUMMARY COMPARISON TABLE
    # ===========================

    st.write(f"""
    ### 🎯 Summary

    With settlement price **Fᵀ = {FT_user}** and initial futures price **F₀ = {F0}**, both hedgers achieve the same outcome:  
    **Long Hedge (buyer):** futures gains/losses offset changes in the purchase price →  **effective cost ≈ F₀**  
    **Short Hedge (seller):** futures gains/losses offset changes in the sales price →   **effective revenue ≈ F₀**         
    **Bottom line:** Futures remove price uncertainty by converting the unknown future spot price **Fᵀ** into the locked-in price **F₀**.
    """)




# =======================================
# 2️⃣ STOCK & COMMODITY FUTURES
# =======================================

elif section == "Stock and Commodity Futures":

    # =======================================
    # 📈 STOCK & COMMODITY FUTURES — INTRO
    # =======================================

    st.title("📈 Stock & Commodity Futures")

    st.write("""
    In the **Basics & Payoffs** section, we introduced how futures contracts work, how they are marked to 
    market, and how traders and firms use them for **hedging** and **speculation**.  
    We also explored how long and short futures positions generate gains and losses depending on 
    movements in the settlement price.

    What we *did not* cover yet is **how the futures price itself is determined**.

    This chapter focuses on the **pricing** of stock and commodity futures using the 
    **cost-of-carry framework**, which links today’s spot price to the futures price through:

    - financing costs (interest rates),  
    - income from holding the asset (dividends or convenience yield),  
    - and costs of holding the asset (storage, insurance, transport).

    We begin with the simplest building block — a **non-dividend paying stock** — and then extend the 
    pricing logic to **dividend-paying equities** and **physical commodities**.
    """)

    st.markdown("---")

    # =====================================================
    # SECTION 1 — NON-DIVIDEND STOCK FUTURE (DERIVATION)
    # =====================================================
    st.subheader("1️⃣ Futures on a Non-Dividend Paying Stock")
    st.write("""
            For a stock with **no dividends**, the cost of carry comes only from **financing the purchase**. We derive the futures price using a **no-arbitrage replication argument**""")

    colA, colB = st.columns([1.1, 1])

    with colA:
        st.write("#### 🔹 Construct Two Equivalent Strategies")
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown(""" 
            **Strategy A — Buy the stock today**  
            - Pay \( S₀ \) now  
            - Hold the stock until maturity \( T \)
            """)

        with col2:
            st.markdown("""
            **Strategy B — Enter a futures contract**  
            - Pay **nothing today**  
            - Agree to buy the stock at \( F₀ \) at time \( T \)
            """)


        #### 🔹 Step 2 — No-Arbitrage Condition  
        st.write("If the two strategies are economically identical, their **future value must match**:")

        st.latex(r"S_0 \, *e^{r*T} = F_0")

        st.markdown("""
        This reflects that buying the stock today ties up capital that could otherwise earn the **risk-free rate**: r.


        #### 🔹 Resulting Pricing Formula
        """)
        st.latex(r"F_0 = S_0 \, *e^{r*T}")
        st.markdown("""
        **Intuition:**  
        The futures price equals the **fully-financed cost** of holding the stock until maturity.  
        No dividends → no adjustments → pure cost-of-carry.
        """)


    with colB:
    # GRAPH — simple non-dividend futures curve linked to sidebar inputs
        T_vals = np.linspace(0.1, 10.0, 100)  # max maturity same as sidebar
        F_vals = S0 * np.exp(r * T_vals)      # use S0 and r from sidebar

        fig, ax = plt.subplots()
        ax.plot(T_vals, F_vals, color="#1a73e8", linewidth=2)
        ax.set_title("Non-Dividend Stock Futures Curve", fontsize=12)
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Futures Price")
        ax.grid(alpha=0.25)
        st.pyplot(fig)

    st.markdown("---")


    # =====================================================
    # SECTION 2 — STOCK FUTURES: DIVIDEND VS NON-DIVIDEND
    # =====================================================
    st.subheader("2️⃣ Futures on Dividend-Paying Stocks")

    st.write("""
    For **stocks that pay dividends**, holding a futures contract means you **do not receive the dividends**. This reduces the futures price compared to a non-dividend stock.  

    We can derive the futures price using the same **no-arbitrage replication argument**:
    """)

    colA, colB = st.columns([1.1, 1])

    with colA:
        st.write("#### 🔹 Construct two Equivalent Strategies")

        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown(""" 
            **Strategy A — Buy the stock today**  
            - Pay \( S₀ \) now  
            - Hold the stock until maturity \( T \)  
            - Collect dividends along the way (total present value = \( PV(\text{Dividends}) \))
            """)

        with col2:
            st.markdown("""
            **Strategy B — Enter a futures contract**  
            - Pay **nothing today**  
            - Agree to buy the stock at \( F₀ \) at time \( T \)  
            - Receive **no dividends** during the holding period
            """)

        #### 🔹 Step 2 — No-Arbitrage Condition  
        st.write("""
        For these strategies to be economically equivalent, the futures price must **adjust for lost dividend income**:

        \[
        S₀ \, e^{rT} - PV(\text{Dividends}) = F₀
        \]

        Using a **continuous dividend yield** \( q \), this simplifies to:
        """)

        st.latex(r"F₀ = S₀ \, e^{(r - q) T}")

        st.markdown("""
        **Intuition:**  
        - Futures prices are lower than the non-dividend case because the holder **misses out on dividends**.  
        - Higher dividends → lower futures price.  
        - Higher interest rates → higher futures price (financing cost effect).
        """)

    with colB:
        # GRAPH — Dividend vs Non-Dividend Futures Curve
        T_vals = np.linspace(0.1, 10.0, 100)
        F_non = S0 * np.exp(r * T_vals)
        F_div = S0 * np.exp((r - div_yield) * T_vals)

        fig, ax = plt.subplots()
        ax.plot(T_vals, F_non, color="#1a73e8", linewidth=2, label="Non-Dividend")
        ax.plot(T_vals, F_div, color="#ff6b6b", linewidth=2, linestyle="--", label="Dividend-Paying")
        ax.set_title("Futures Price: Dividend vs Non-Dividend Stock")
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Futures Price")
        ax.grid(alpha=0.25)
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")



    # =====================================================
    # SECTION 3 — BASIS RISK (GRAPH | SHORT TEXT)
    # =====================================================
    st.subheader("3️⃣ Basis Risk")

    colA, colB = st.columns([1, 1.4])

    with colA:
        S_t = st.number_input("Spot Price Today Sₜ", value=100.0)
        F_t = st.number_input("Futures Price Today Fₜ", value=102.0)

        basis = S_t - F_t
        st.metric("Basis (Sₜ - Fₜ)", f"{basis:,.2f}")

        st.write("""
        - Basis fluctuates due to inventory, seasonality,  
          convenience yields, or dividend changes.  
        - **Hedge fails** when spot & futures do not track perfectly.
        """)

    with colB:
        # GRAPH — Simulated basis evolution
        t = np.linspace(0, 1, 50)
        basis_sim = basis + np.sin(6*t) * 2  # artificial illustration

        fig2, ax2 = plt.subplots()
        ax2.plot(t, basis_sim)
        ax2.axhline(0, linestyle="--")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Basis")
        ax2.set_title("Basis Fluctuation Illustration")
        st.pyplot(fig2)

    st.markdown("---")

    # =====================================================
    # SECTION 4 — ROLLOVER RISK (GRAPH)
    # =====================================================
    st.subheader("4️⃣ Rollover Risk (Contango vs Backwardation)")

    colX, colY = st.columns([1, 1.4])

    with colX:
        st.write("""
        When rolling from near-month to next-month futures:
        - **Contango** → Roll *costs* money  
        - **Backwardation** → Roll *earns* money  

        ETFs like **USO** and **UNG** are heavily affected by this.
        """)

        slope = st.slider("Term Structure Slope", -10.0, 10.0, 3.0)

    with colY:
        t = np.linspace(0.1, 3, 30)
        curve = S0 * (1 + slope/100 * t)

        fig3, ax3 = plt.subplots()
        ax3.plot(t, curve)
        ax3.set_xlabel("Maturity")
        ax3.set_ylabel("Futures Price")
        ax3.set_title("Term Structure (Contango / Backwardation)")
        st.pyplot(fig3)

    st.markdown("---")

    # =====================================================
    # SECTION 5 — METALLGESELLSCHAFT (CONDENSED)
    # =====================================================
    st.subheader("5️⃣ Real-World Case: Metallgesellschaft (1993)")

    colM1, colM2 = st.columns([1, 1.4])

    with colM1:
        st.write("""
        **Why the hedge failed:**
        - Long-term fixed-price sales hedged with short-term futures  
        - Sharp backwardation → temporary losses  
        - Massive **margin calls**  
        - Hedge forced to unwind prematurely  
        """)

    with colM2:
        # A simple loss-then-recover curve to illustrate MtM losses
        t = np.linspace(0, 1, 200)
        pnl = -20*np.exp(-6*t) + 20*(1 - np.exp(-4*t))

        fig4, ax4 = plt.subplots()
        ax4.plot(t, pnl)
        ax4.axhline(0, linestyle="--")
        ax4.set_title("Mark-to-Market Loss vs Economic Value")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("PnL")
        st.pyplot(fig4)

    st.info("""
    **Lesson:**  
    Even a *perfect long-term hedge* can fail if the hedger cannot survive short-term 
    mark-to-market cash requirements.
    """)

    st.markdown("---")
















# =======================================
# 3️⃣ FX & INTEREST RATE FUTURES
# =======================================

elif section == "FX & Interest Rate Futures":
    st.header("🌍 FX Futures")
    st.latex(r"""
    F_0 = S_0 \cdot e^{(r_{domestic} - r_{foreign})T}
    """)

    st.sidebar.header("FX Futures Inputs")
    S0_fx = st.sidebar.number_input("Spot FX Rate", 0.1, 10.0, 1.10)
    r_dom = st.sidebar.slider("Domestic Rate (%)", 0.0, 15.0, 5.0) / 100
    r_for = st.sidebar.slider("Foreign Rate (%)", 0.0, 15.0, 3.0) / 100
    T_fx = st.sidebar.slider("Maturity (years)", 0.1, 5.0, 1.0)

    FX_future = S0_fx * np.exp((r_dom - r_for) * T_fx)

    st.write(f"**FX Futures Price:** {FX_future:.4f}")

    # Interest Rate Futures
    st.header("📉 Interest Rate Futures (Eurodollar / Treasury)")
    st.write("""
        Interest rate futures reflect the market’s expectation for future interest rates.

        Example:  
        **Eurodollar future price = 100 − implied 3-month LIBOR**
    """)

    rate_future_price = st.sidebar.slider("Implied Rate (%)", 0.0, 10.0, 5.0)
    st.write(f"**Eurodollar Futures Price = 100 − rate = {100 - rate_future_price:.2f}**")













# =======================================
# 4️⃣ INDEX & VIX FUTURES
# =======================================

elif section == "Index & VIX Futures":
    st.header("📈 Index Futures")
    st.write("""
        For stock index futures, the pricing is:
    """)
    st.latex(r"""
    F_0 = S_0 e^{(r - d)T}
    """)
    st.write("""
        Where:  
        - \( d \) = dividend yield  
    """)

    st.sidebar.header("Index Inputs")
    S0_idx = st.sidebar.number_input("Index Spot Level", 100, 10000, 5000)
    div_yield = st.sidebar.slider("Dividend Yield (%)", 0.0, 5.0, 1.5) / 100
    r_idx = st.sidebar.slider("Risk-free Rate (%)", 0.0, 10.0, 4.0) / 100
    T_idx = st.sidebar.slider("Maturity (years)", 0.1, 3.0, 1.0)

    F_idx = get_future_price_spot_costcarry(S0_idx, r_idx, 0, div_yield, T_idx)

    st.write(f"**Index Futures Price:** {F_idx:,.2f}")

    # ---------------------------
    # VIX FUTURES
    # ---------------------------
    st.header("⚡ VIX Futures")
    st.write("""
        The VIX index is not directly tradable.  
        VIX futures reflect the **market expectation of future volatility**, not spot VIX.
    """)

    # Fetch live VIX + futures from Yahoo
    try:
        vix = yf.Ticker("^VIX")
        vix_price = vix.history(period="1d")["Close"][-1]
        st.write(f"**Current VIX:** {vix_price:.2f}")
    except:
        st.write("⚠ Could not load VIX data.")

    st.write("""
        VIX futures often trade at a **premium** (contango) or **discount** (backwardation)  
        depending on market stress levels.
    """)

