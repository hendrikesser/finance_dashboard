import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ===========================
# Page Title & Theme
# ===========================
st.set_page_config(page_title="Futures Dashboard", layout="wide")
st.title("📈 Futures Dashboard")


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
        "Index & VIX Futures",
        "Arbitrage Strategies"
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
    F0 = st.sidebar.slider("Initial Futures Price F₀", min_value=1, max_value=100, value=50)
    payoff_qty = st.sidebar.slider("Quantity (Contract Size)", 1, 100, 10)
    FT_user = st.sidebar.slider("Settlement Price Fᵀ", min_value=1, max_value=100, value=60)

    # Number of days for MtM simulation
    st.sidebar.markdown("---")
    st.sidebar.subheader("Mark-to-Market Parameters")
    days = st.sidebar.slider("Number of Days to Simulate", min_value=1, max_value=30, value=8)

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
    st.sidebar.header("Stock Futures Inputs")
    S0 = st.sidebar.number_input("Spot Price S₀", 1, 5000, 100)
    r = st.sidebar.slider("Risk-free rate (%)", 0.0, 20.0, 3.0) / 100
    div_yield = st.sidebar.slider("Dividend Yield (%)", 0.0, 10.0, 2.0) / 100
    T = st.sidebar.slider("Maturity (years)", 0.1, 10.0, 1.0)

    st.sidebar.header("Commodity Futures Inputs")
    storage = st.sidebar.slider("Storage Cost (%)", 0.0, 10.0, 1.0) / 100
    conv_yield = st.sidebar.slider("Convenience Yield (%)", 0.0, 10.0, 1.0) / 100
    st.sidebar.write("**Note**: use T & r from above")

    st.sidebar.header("Basis Risk Parameters")
    days_slider = st.sidebar.slider("Number of Days to Simulate", 3, 15, 7)

    st.sidebar.header("Rollover Parameters")
    S = float(st.sidebar.slider("Initial Spot Price S₀", 1, 1000, 100))
    qty = int(st.sidebar.slider("Quantity to Hedge (barrels)", 1, 1000, 100))
    periods = int(st.sidebar.slider("Number of Rollovers (periods)", 1, 10, 3))
    btn = st.sidebar.button("Generate Random Future Spot Path")

    F0_calc = get_future_price_spot_costcarry(S0, r, storage, div_yield, T)
   
   


elif section == "FX & Interest Rate Futures":
    st.sidebar.header("FX Input Parameters")

    S0_fx = st.sidebar.slider("Spot rate S₀ (USD per 1 foreign unit)", 1, 1000, 100)
    r_usd = st.sidebar.slider("USD risk-free rate r_USD (%)", 0.0, 20.0, 5.0) / 100
    r_for = st.sidebar.slider("Foreign risk-free rate r_foreign (%)", 0.0, 20.0, 3.0) / 100
    T_fx = st.sidebar.slider("Maturity T (years)", 0.01, 5.0, 1.0)

elif section == "Index & VIX Futures":
    st.sidebar.header("Index & VIX Futures Inputs")

elif section == "Arbitrage Strategies": 
    st.sidebar.header("Abritrage Inputs")


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



    st.write("#### 🔹 Daily P&L Table and Graph (with F₀ and Quantity)")
    s1, s2 = st.columns(2)
    with s1:
        st.dataframe(df_mtm, use_container_width=True)
        st.write(f"**Total P&L after {days} days:** {cumulative_pl[-1]:.2f}  \nThis is also the economic value of the future contract at the end of day {days}.")
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

        The futures payoff adjusts your final cost so that the **effective price stays near F₀**.   
        Ensuring budget certainty.
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
        st.write("Move the slider to see how different settlement prices affect outcomes (they don't change the effective costs for the long hedge!)")
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
        st.write("Move the slider to see how different settlement prices affect outcomes (they don't change the effective revenues for the short hedge!)")
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

    st.header(" Stock & Commodity Futures")

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
            Similar to commodities (introduced in the basic section, but not yet covered in detail), stocks can also be the underlying asset for futures contracts.
            For a stock with **no dividends**, the cost of carry comes only from **financing the purchase**. We can derive the futures price using a **no-arbitrage replication argument**
            if two different strategies generate the **same payoff at maturity**, they must have the **same value today**.""")

    colA, colB = st.columns([1.1, 1])

    with colA:

        # ------------------ Replication Logic ------------------
        st.write("### 🔹 Replication Approach: Construct Two Equivalent Strategies")

        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("""
            **Strategy A — Buy the stock today**  
            - Pay the spot price \( S_0 \) today  
            - Hold the stock until time \( T \)  
            - **Payoff at \( T \):** receive the stock worth \( S_T \)
            """)

        with col2:
            st.markdown("""
            **Strategy B — Enter a long futures contract**  
            - Pay **nothing today**  
            - Commit to buy the stock at the futures price \( F(T) \) at time \( T \)  
            - **Payoff at \( T \):** receive the same stock, but pay \( F(T) \)
            """)

        st.markdown("""
        Even though the cash flows today differ, **both strategies deliver the same stock at time \(T\)**.  
        To avoid arbitrage, the **future values** of both strategies must be equal.
        """)

        # Pricing Equation
        st.write("### 🔹 No-Arbitrage Condition")
        st.latex(r"S_0 \, e^{rT} = F(T)")

        st.markdown("""
        Buying the stock today ties up capital that could earn the **risk-free rate** \( r \).  
        Thus, the future value of purchasing the stock must equal the futures price.
        """)

     
    
    with colB:
        # ================================
        # Graph: Futures curve
        # ================================
        T_vals = np.linspace(0.1, 10.0, 100)
        F_vals = S0 * np.exp(r * T_vals)

        F_T = S0 * np.exp(r * T)

        fig, ax = plt.subplots()
        ax.plot(T_vals, F_vals, linewidth=2, label="Futures Curve")
        ax.plot(T, F_T, 'o', markersize=8, label=f"F(T={T}) = {F_T:.2f}")

        ax.set_title("Non-Dividend Stock Futures Curve", fontsize=12)
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Futures Price")
        ax.grid(alpha=0.25)
        ax.legend()
        st.pyplot(fig)
        

    

        st.markdown(
            "<h4 style='text-align: center;'>🔍 Futures Pricing Formula</h4>",
            unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Futures Pricing Formula:**  
            F(T) = S_0 \, e^(rT)
            """)

        with col2:
            st.markdown(f"""
            **Calculation:**  
            F({T}) = {S0} * e^{{{r:.3f} * {T}}} = {F_T:.2f}
       
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.write("### 🔹 Resulting Pricing Formula")
        st.latex(r"F(T) = S_0 \, e^{rT}")

        st.markdown("""
        **Where:**  
        - \( F(T) \): fair futures price for delivery at time \( T \)  
        - \( S_0 \): current spot price  
        - \( r \): risk-free interest rate  
        - \( T \): time to maturity (in years)""")

    with col2:
        st.markdown("""
            ### 🔹 Economic Intuition
            - Forward/futures price is the **cost of buying the stock and financing it until \( T \)**.  
            - The stock’s uncertainty **does not matter** — its risk is already priced into \( S_0 \).  
            - With no dividends, there are **no adjustments**, so the futures price is simply the **fully financed spot price**.
            """)


    st.markdown("---")


    # =====================================================
    # SECTION 2 — STOCK FUTURES: DIVIDEND VS NON-DIVIDEND
    # =====================================================
    st.subheader("2️⃣ Futures on Dividend-Paying Stocks")

    st.write("""
    Analogously to the non-dividend case, we can write Futures on dividend paying stocks. Holding a futures contract means you **do not receive the dividends**. This reduces the futures price compared to a non-dividend stock, as you are missing out on that income.

    We can derive the futures price using the same **no-arbitrage replication argument**:
    """)

    colA, colB = st.columns([1.1, 1])

    with colA:
        st.subheader("🔹 Two Equivalent Strategies for Dividend-Paying Stocks")

        # Use columns for side-by-side comparison
        strat_col1, strat_col2 = st.columns(2)

        with strat_col1:
            st.markdown("""
            **Strategy A — Buy the Stock Today**  
            - Pay \(S_0\) upfront  
            - Hold the stock until maturity \(T\)  
            - Receive dividends during the holding period  
            """)

        with strat_col2:
            st.markdown("""
            **Strategy B — Use a Futures Contract**  
            - Pay nothing today  
            - Agree to buy the stock at \(F_0\) at maturity \(T\)  
            - Do **not** receive dividends
            """)

        # No-arbitrage explanation
        st.markdown("""
        **No-Arbitrage Condition**  
        To prevent arbitrage, the futures price must adjust for the dividends missed by the futures holder:""")

        st.latex(r"F_0 = S_0 \cdot e^{r T} - PV(\text{Dividends})")

        st.markdown("""
        If we assume a **continuous dividend yield** \(q\), this simplifies to:
        """)
        st.latex(r"F_0 = S_0 \, e^{(r - q)*T}")

    with colB:
        # GRAPH — Dividend vs Non-Dividend Futures Curve
        T_vals = np.linspace(0.1, 10.0, 100)
        F_non = S0 * np.exp(r * T_vals)
        F_div = S0 * np.exp((r - div_yield) * T_vals)

        # Calculate futures price at the selected maturity T
        F_non_T = S0 * np.exp(r * T)
        F_div_T = S0 * np.exp((r - div_yield) * T)

        fig, ax = plt.subplots()
        ax.plot(T_vals, F_non, color="#1a73e8", linewidth=2, label="Non-Dividend")
        ax.plot(T_vals, F_div, color="#ff6b6b", linewidth=2, linestyle="--", label="Dividend-Paying")
        
        # Highlight the selected maturity with orange dots
        ax.plot(T, F_non_T, 'o', color='orange', markersize=8, label=f"Non-Dividend F(T={T}) = {F_non_T:.2f}")
        ax.plot(T, F_div_T, 'o', color='darkorange', markersize=8, label=f"Dividend F(T={T}) = {F_div_T:.2f}")
        
        ax.set_title("Futures Price: Dividend vs Non-Dividend Stock")
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Futures Price")
        ax.grid(alpha=0.25)
        ax.legend()
        st.pyplot(fig)

        # Display formulas and calculated futures prices
        st.markdown(
            "<h4 style='text-align: center;'>🔍 Futures Pricing Formulas</h4>",
            unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Non-Dividend Stock:**  
            F(T) = S₀ × e^(r × T)  
            **Dividend-Paying Stock:**  
            F(T) = S₀ × e^((r - q) × T)
            """)
        with col2:
            st.markdown(f"""
            **Calculation**: F({T}) = {S0} × e^({r} × {T}) = **{F_non_T:.2f}**\n
            **Calculation**: F({T}) = {S0} × e^(({r} - {div_yield:.3f}) × {T}) = **{F_div_T:.2f}**
            """)

    st.markdown("""### 🔹 Economic Intuition""")
    col1, col2 = st.columns(2)
    with col1:
        st.write(""" 
        - Futures prices are lower than the non-dividend case because the holder **does not receive dividends**, reducing the attractiveness of holding the futures versus the underlying stock.  
        - **Higher dividends → lower futures price**, as the stock becomes more valuable due to expected payouts.""")
    with col2:
        st.write("""
        - **Higher interest rates → higher futures price**, reflecting the increased cost of financing the underlying stock.""")


    st.markdown("---")



    # =====================================================
    # SECTION 3 — COMMODITY FUTURES
    # =====================================================
    st.subheader("3️⃣ Commodity Futures")

    st.markdown("""
    In the **introduction chapter**, we mentioned commodities as one underlying asset class of futures contracts.  
    Their pricing also follows the **cost-of-carry framework**, but with additional components unique to physical goods:

    - **Storage, insurance, and transport costs** \(u\)  
    - **Convenience yield** \(y\) — benefits from holding the commodity physically  


    This section derives the pricing formula using the no-arbitrage principle and visualizes how commodity futures prices evolve across maturities.
    """)

    colA, colB = st.columns([1.1, 1])

    # LEFT COLUMN — THEORY & INTUITION (REWORKED)
    with colA:

        st.write("### 🔹 Replication Approach: Commodity Futures")

        st.markdown("""
        Similar to stocks, we can derive commodity futures prices using a **no-arbitrage replication argument**.  
        Imagine two strategies that deliver the **same commodity at time \(T\)**:
        """)

        strat_col1, strat_col2 = st.columns(2)

        with strat_col1:
            st.markdown("""
            **Strategy A — Buy the Commodity Today**  
            - Pay the spot price \(S_0\) upfront  
            - Store, insure, and transport the commodity until maturity  
            - Receive the commodity at time \(T\)  
            - Total cost grows at the risk-free rate and includes storage costs \(u\)  
            """)

        with strat_col2:
            st.markdown("""
            **Strategy B — Enter a Long Futures Contract**  
            - Pay nothing today  
            - Agree to receive the commodity at futures price \(F(T)\) at time \(T\)  
            - Avoid storage costs but also miss the **convenience yield** \(y\) from holding the commodity
            """)

        st.markdown("""
        Both strategies result in **owning the commodity at time \(T\)**. To avoid arbitrage, the future value of Strategy A must equal the payoff of Strategy B at maturity.
        Strategy A grows at the risk-free rate \(r\), plus storage cost \(u\), minus convenience yield \(y\), giving the futures price:
        $$
        F(T) = S₀ × e^{(r + u - y) × T}  
        $$
        """)

    # RIGHT COLUMN — GRAPH + FORMULA BREAKDOWN
    with colB:

        st.write("### 🔹 Commodity Futures Curve Example")

        # Maturity axis
        T_vals = np.linspace(0.1, 10.0, 100)

        # Futures curve
        F_commodity = S0 * np.exp((r + storage - conv_yield) * T_vals)

        # Price at selected maturity
        F_commodity_T = S0 * np.exp((r + storage - conv_yield) * T)

        # Plot
        fig, ax = plt.subplots()
        ax.plot(T_vals, F_commodity, linewidth=2, label="Commodity Futures")
        ax.plot(T, F_commodity_T, 'o', color='darkorange', markersize=8,
                label=f"F(T={T}) = {F_commodity_T:.2f}")

        ax.set_title("Commodity Futures Price Curve")
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Futures Price")
        ax.grid(alpha=0.25)
        ax.legend()
        st.pyplot(fig)

        # Centered heading above formulas (matching your other sections)
        st.markdown(
            "<h4 style='text-align:center;'>🔍 Futures Pricing Formula</h4>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Commodity Futures:**  
            F(T) = S₀ × e^{(r + u - y) × T}  
            """)

        with col2:
            st.markdown(
                fr"""
                **Calculation:**  
                F({T}) = {S0} * e^{{({r:.3f} + {storage:.3f} - {conv_yield:.3f}) * {T}}}
                = {F_commodity_T:.2f}
                """)

    st.markdown("""
        ### 🔹 Economic Intuition""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""

        - **Higher storage costs \(u\) → futures price rises**  
        More resources are needed to store, insure, and transport the commodity, increasing the cost of carry.

        - **Higher convenience yield \(y\) → futures price falls**  
        Holding the physical commodity provides benefits (e.g., production security), making futures less attractive.""")
    with col2:
        st.markdown("""
        - **Longer maturities amplify effects**  
        Costs and yields accumulate over time, so longer-dated futures are more sensitive to \(u\) and \(y\).
        """)



    st.markdown("---")

    # =====================================================
    # SECTION 4 — BASIS RISK (Dynamic Simulation)
    # =====================================================
    st.subheader("4️⃣ Basis Risk")

    st.markdown("""
    So far our discussion has assumed **perfect hedging** with futures contracts. 
    In reality there are several issues that might arise: 

    1. The asset being hedged may differ from the futures contract.
    2. The agents might not know the exact timing of selling/buying the underlying.
    3. The futures contract may expire before the underlying transaction occurs.

    These uncertainties are often all combined in the term **Basis risk**, which is defined as the difference between the **spot price** of the asset being hedged and the **futures price** of the contract used:

    $$
    Basis = S(t) - F(t)
    $$

    """)

    # --------------------------
    # Slider Inputs for Simulation
    # --------------------------
    st.write("In order to illustrate basis risk, we simulate spot and futures prices over time with imperfect correlation. Use commodity and basis risk parameters in the sidebar.")
    F0_basis = get_future_price_spot_costcarry(S0, r, storage, conv_yield, T)
    S0_slider = S0
    F0_slider = F0_basis

    # --------------------------
    # Simulate Price Paths
    # --------------------------
    np.random.seed(42)
    spot_changes = np.random.normal(0, 0.5, days_slider).cumsum()
    spot_prices = S0_slider + spot_changes

    # Futures converging to spot at maturity
    fut_changes = np.random.normal(0, 0.45, days_slider).cumsum()
    raw_fut_prices = F0_slider + fut_changes

    # Linear adjustment to force convergence at maturity
    fut_prices = raw_fut_prices + (spot_prices[-1] - raw_fut_prices[-1]) * np.linspace(0, 1, days_slider)

    basis = spot_prices - fut_prices

    basis_df = pd.DataFrame({
        "Day": np.arange(1, days_slider+1),
        "Spot Price": np.round(spot_prices, 2),
        "Futures Price": np.round(fut_prices, 2),
        "Basis": np.round(basis, 2)
    })
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔄 Simulate Basis Over Time")
        st.dataframe(basis_df, hide_index=True)
    with col2:
        st.markdown("### 📈 Spot, Futures, and Basis Over Time")

        fig, ax1 = plt.subplots(figsize=(7,4))

        # Spot and futures
        ax1.plot(basis_df["Day"], basis_df["Spot Price"], label="Spot Price", color="#1a73e8", linewidth=2)
        ax1.plot(basis_df["Day"], basis_df["Futures Price"], label="Futures Price", color="#ff6b6b", linewidth=2)
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Price")
        ax1.set_title("Spot vs Futures Price & Basis Over Time")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="upper left")

        # Basis as secondary axis
        ax2 = ax1.twinx()
        ax2.plot(basis_df["Day"], basis_df["Basis"], label="Basis", color="#ffa500", linestyle="--", linewidth=2)
        ax2.set_ylabel("Basis")
        ax2.legend(loc="upper right")

        st.pyplot(fig)

    st.write("With the table and graph we can see how basis risk evolves over time due to imperfect correlation between spot and futures prices, eventually converging at maturity.")

    # --------------------------
    # Numerical Example
    # --------------------------
    st.markdown("### Example: Calculating Basis")

    S1, F1 = spot_prices[0], fut_prices[0]
    S2, F2 = spot_prices[-1], fut_prices[-1]
    b1, b2 = S1-F1, S2-F2
    effective_price = S2 + (F1-F2)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **At Initiation (Day 1):**  
        - Spot Price S₁ = {S1:.2f}  
        - Futures Price F₁ = {F1:.2f}  
        - Basis b₁ = S₁ - F₁ = {b1:.2f}  
        """)
    with col2:
        st.markdown(f"""
    **Takeaways:**  
    - Basis fluctuates due to imperfect correlation of spot and futures.  
    - Short hedge benefits if basis **strengthens** (b₂ ↑), worsens if basis **weakens** (b₂ ↓).  
    - Cross-hedging increases uncertainty, adding to basis risk.
    """)

    st.markdown("---")

    # =====================================================
    # SECTION 5 — ROLLOVERS (Stack & Roll)
    # =====================================================
    st.subheader("5️⃣ Rollovers and Maturity Mismatch")
    st.write("""
    When the exposure (e.g., selling oil in 14 months) lasts **longer** than the available futures contract maturities, the hedger must **roll** the hedge forward.
    This is known as **stack and roll**:""")
    colA, colB = st.columns(2)
    with colA:
        st.write("""
    - Short the nearest liquid futures contract  
    - Close it when it expires or becomes illiquid""")
    with colB:
        st.write("""  
    - Open a later-dated contract  
    - Repeat until the date when the underlying is bought/sold
    """)

    # -------------------------
    # Initialize
    # -------------------------
    rng = np.random.default_rng()
    current_spot = S
    records = []

    for y in range(1, periods + 1):
        # Open futures price using no-arbitrage formula with storage and convenience yield
        F_open = current_spot * np.exp((r + storage - conv_yield) * 1)  # 1-year futures

        # Random closing future price (simulate market movement)
        F_close = F_open * (1 + rng.normal(0, 0.01))  # ~1% random annual move
        gain = F_open - F_close

        records.append([y, round(current_spot, 2), round(F_open, 2), round(F_close, 2), round(gain, 2)])

        # Update spot for next rollover (simulate market movement)
        current_spot = F_close * (1 + rng.normal(0, 0.01))


   # -------------------------
    # Build DataFrame
    # -------------------------
    df = pd.DataFrame(records, columns=["Year", "Spot Start", "Open Future Price", "Close Future Price", "Gain/Rollover"])

    # Add a summary row with total rollover costs
    total_gain_per_barrel = df["Gain/Rollover"].sum()
    df.loc[len(df)] = ["Total", np.nan, np.nan, np.nan, round(total_gain_per_barrel, 2)]

    # -------------------------
    # Display table
    # -------------------------
    st.markdown("### Annual Rolling Hedge Table")
    st.write("This table summarizes the annual rollovers of the futures hedge over the specified periods for only one barrel. The total gain/loss from rolling the futures contracts is shown in the last row.")

    # Select numeric columns except Year
    numeric_cols = df.columns.drop("Year")

    # Style numeric columns with 2 decimals
    styled_df = df.style.format({col: "{:.2f}" for col in numeric_cols})

    # Display without index
    st.dataframe(styled_df, use_container_width=True, hide_index=True)



   # -------------------------
    # Summary
    # -------------------------
    total_gain_per_barrel = df.loc[df["Year"] != "Total", "Gain/Rollover"].sum()
    total_gain_dollars = total_gain_per_barrel * qty
    avg_gain_per_year = df.loc[df["Year"] != "Total", "Gain/Rollover"].mean()
    min_gain = df.loc[df["Year"] != "Total", "Gain/Rollover"].min()
    max_gain = df.loc[df["Year"] != "Total", "Gain/Rollover"].max()
    final_spot = df.loc[df.index[-2], "Close Future Price"]  # last real row

    st.markdown("### 📊 Hedge Summary (At a Glance)")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- **Initial Spot Price S₀:** ${S:.2f}")
        st.write(f"- Total Gain/Loss per Barrel: ${total_gain_per_barrel:.2f}")
        st.write(f"- **Total Gain/Loss for {qty:,} barrels:** ${total_gain_dollars:,.2f}")
    with col2:
        st.write(f"Final Spot Price after {periods} rollovers (at maturity equals Future Price): ${final_spot:.2f}")
        st.write(f"Average Gain/Loss per Year: ${avg_gain_per_year:.2f}")
        st.write(f"**Total number of rollovers:** {periods-1}")



    st.markdown("---")

    # =====================================================
    # SECTION 6 — HEDGING RISK: METALLGESELLSCHAFT CASE
    # =====================================================
    st.subheader("6️⃣ Hedging Risk — The Metallgesellschaft Case")

    st.write("""
    In the 1990s, **Metallgesellschaft AG** (Germany) attempted a long-dated oil hedge using **short-term futures contracts**.  
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔹 What happened")
        st.write("""
        - Hedged long-term fixed-price supply contracts with **short-dated futures rolled monthly**  
        - Oil prices fell, leading to **margin calls and short-term cash outflows**  
        - Expected gains on long-term contracts did not materialize in time to cover losses  
        - MG closed all positions → loss of **$1.33 billion**
        """)

    with col2:
        st.markdown("### 🔹 Lessons Learned")
        st.write("""
    - Hedging with **short-term futures for long-term exposure** carries significant **rollover and liquidity risk**  
    - **Basis risk** can amplify losses if timing mismatches occur  
    - Hedgers must **plan cash flows, monitor liquidity, and manage rollover strategy**
    """)

















# =======================================
# 3️⃣ FX & INTEREST RATE FUTURES
# =======================================


elif section == "FX & Interest Rate Futures":

    # ============================================================
    # 🌍 FX FUTURES — INTRODUCTION
    # ============================================================
    st.header("🌍 FX Futures & Forward Pricing")

    st.write("""
    Foreign exchange (FX) futures allow traders and institutions to buy or sell a foreign 
    currency at a predetermined price on a future date. Just like other futures contracts, 
    FX futures are standardized, exchange-traded, marked-to-market daily, and carry very 
    low counterparty risk.

    What makes FX futures unique is that *a currency itself generates a known yield* — the 
    risk-free interest rate of that country. This leads to one of the cleanest applications 
    of the cost-of-carry model: **Covered Interest Rate Parity (CIP)**.
    """)

    st.markdown("---")

    # ============================================================
    # 🔹 KEY CHARACTERISTICS
    # ============================================================
    st.subheader("🔹 Key Characteristics of FX Futures")
    colA, colB = st.columns(2)
    with colA: 
        st.write("""
        - **Standardized**: Exchanges (CME, ICE) define contract size and delivery dates.  
        - **Two sides**:  
        - Long → agrees to buy foreign currency  
        - Short → agrees to sell foreign currency  
        - **Daily mark-to-market**: Gains/losses settled every day.""")
    with colB:
        st.write("""  
        - **Zero initial value**: Futures always start with value = 0.  
        - **Low credit risk**: Clearinghouse guarantees performance.  
        - **Cash or physical settlement**: Depends on the contract.  
        - **Highly liquid**: Ideal for hedging, arbitrage, and speculation.
        - "**Convention note:** In this dashboard, we quote FX rates as _USD per 1 unit of foreign currency_ — e.g. 0.62 USD/AUD means 1 Australian dollar = 0.62 US dollars.  
        """)

    st.markdown("---")

    # ============================================================
    # 🌟 HOW FX FUTURES ARE PRICED — NO-ARBITRAGE DERIVATION
    # ============================================================
    st.subheader("📘 How the FX Forward/Futures Formula Is Derived")

    st.write("""
    To price FX forwards or futures, we use **the same no-arbitrage replication logic** 
    used for stock and commodity futures. But here, the key insight is:

    #### Holding 1 unit of foreign currency earns the **foreign risk-free rate**
    So the foreign currency behaves like an investment asset with a *known yield* (just like a dividend-paying asset):  
    the foreign interest rate \( r_f \). We now compare two strategies that must deliver the **same number of USD at time T**.
    """)

    col1, col2 = st.columns(2)

    # ----------------------------------------
    # Strategy A — Invest in USD
    # ----------------------------------------
    with col1:
        st.markdown("""
        ### **Strategy A — Invest in USD Today**

        1. Start with **1 USD**  
        2. Invest it at the U.S. risk-free rate \( r_{\text{USD}} \)  
        3. The investment grows to:
        """)

        st.latex(r"1 \cdot e^{\, r_{\text{USD}} \, *T}")


    # ----------------------------------------
    # Strategy B — Convert & Hedge
    # ----------------------------------------
    with col2:
        st.markdown("""
        ### **Strategy B — Convert to Foreign Currency & Hedge**

        1. Convert **1 USD** into foreign currency: 1/S_0  
        (Where S_0 is the exchange rate expressed in USD)
        """) 

        st.markdown("2. Invest this at the foreign rate \( r_f \)")

        st.markdown("3. At time \(T\), you have:")
        st.latex(r"\frac{1}{S_0} e^{r_{\text{f}} *T}")

        st.markdown("4. Lock in a forward to convert foreign currency to USD at \(F_0\)")

        st.markdown("5. Dollar value at \(T\) becomes:")
        st.latex(r"F_0 \cdot \frac{1}{S_0} e^{r_{\text{f}} *T}")


    # ----------------------------------------
    # Equating both strategies → CIP
    # ----------------------------------------
    st.write("""
    In the absence of arbitrage, both strategies must yield **the same USD amount at T**:
    """)

    st.latex(r"e^{r_{\text{USD}} T} = F_0 \cdot \frac{1}{S_0} e^{r_{\text{for}} T}")

    st.write("Solving for the fair forward price \(F_0\):")

    st.latex(r"F_0 = S_0 \, e^{(r_{\text{USD}} - r_{\text{for}}) T}")

    st.write("This is the famous **Covered Interest Rate Parity (CIP)** condition.")

    st.markdown("---")

    # ============================================================
    # 📈 FX Forward Price (CIP) — Full Interactive Explanation
    # ============================================================

    st.subheader("📘 FX Forward Pricing Using Covered Interest Parity (CIP)")


    # Forward based on CIP
    F0_val = S0_fx * np.exp((r_usd - r_for) * T_fx)

    # Columns for layout
    col1, col2 = st.columns([1, 1.3])

    # ------------------------------------------------------------
    # LEFT COLUMN — EXPLANATION + NUMERICAL TABLE
    # ------------------------------------------------------------
    with col1:

        st.markdown("#### 🌍 **Covered Interest Parity (CIP)**")

        st.write(r"""
    Covered Interest Parity is the **no-arbitrage condition** linking interest rates, spot FX rates,
    and forward FX rates.

    It says that hedged returns must be **equal across currencies**.  
    If you:

    1️⃣ Invest in **USD** at rate \( r_{\text{USD}} \)

    2️⃣ OR convert to foreign currency, invest at \( r_{\text{for}} \),  
    and lock in the forward rate \( F_0 \)

    …you must end up with the **same USD amount at maturity**.

    Otherwise an arbitrage opportunity would exist.

    Mathematically:

    \[
    F_0 = S_0 \, e^{(r_{\text{USD}} - r_{\text{for}}) T}
    \]

    This is the fair **FX forward price** under no arbitrage.
    """)

    # ------------------------------------------------------------
    # RIGHT COLUMN — FORWARD PRICE VS MATURITY GRAPH
    # ------------------------------------------------------------
    with col2:

        st.markdown("#### 📈 **Forward Price vs. Maturity (CIP Forward Curve)**")

        T_range = np.linspace(0.01, 5.0, 300)
        F_curve = S0_fx * np.exp((r_usd - r_for) * T_range)

        fig, ax = plt.subplots()
        ax.plot(T_range, F_curve)
        ax.set_xlabel("Maturity T (years)")
        ax.set_ylabel("Forward Price F₀(T)")
        ax.set_title("Forward Curve Implied by CIP")
        ax.scatter([T_fx],[F0_val], s=8**2, color='orange', marker='o', zorder=5)


        st.pyplot(fig)

    st.markdown("#### 🔢 **Numerical Example**")

    df_example = pd.DataFrame({
            "Variable": ["Spot rate S₀", "USD rate r_USD", "Foreign rate r_foreign", "Maturity T", "Forward price F₀"],
            "Value": [
                f"{S0_fx:.2f} USD/foreign",
                f"{r_usd*100:.2f} %",
                f"{r_for*100:.2f} %",
                f"{T_fx:.2f} years",
                f"{F0_val:.2f} USD/foreign"
            ]
        })
    st.dataframe(df_example, use_container_width=True, hide_index=True)

    st.markdown("---")

 
    @st.cache_data
    def get_fx_spot_rates():
        # FX pairs
        pairs = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDJPY=X", "USDCNY=X"]
        
        # Download all tickers at once
        df_raw = yf.download(tickers=" ".join(pairs), period="2d")["Close"]
        if df_raw.empty:
            return pd.DataFrame(columns=["Currency Pair", "Spot Rate (USD per foreign unit)"])
        
        latest_row = df_raw.tail(1)
        data = {}
        
        for p in pairs:
            if p in latest_row:
                spot = latest_row[p].values[0]
                
                # Determine if inversion is needed
                if p.startswith("USD"):  # USDXXX → invert to get USD per foreign unit
                    spot = 1 / spot
                # XXXUSD → already USD per foreign unit, no change
                
                data[p] = spot
            else:
                data[p] = None  # placeholder if no data
        
        df_spot = pd.DataFrame({
            "Currency Pair": [p.replace("=X","") for p in pairs],
            "Spot Rate (USD per foreign unit)": [data[p] for p in pairs]
        })
        
        return df_spot  # <---- RETURN HERE

    # Fetch FX spot rates
    df_spot = get_fx_spot_rates()

    st.markdown("### 🔎 Reference FX Spot Rates (USD per 1 foreign unit)")
    st.dataframe(df_spot, use_container_width=True, hide_index=True)

    st.markdown("---")


    

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

elif section == "Arbitrage Strategies":
    st.header("Arbitrage Section")