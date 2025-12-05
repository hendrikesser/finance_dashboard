import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.optimize import fsolve
from streamlit_option_menu import option_menu
import yfinance as yf




# ===========================
# Page Title & Theme
# ===========================
st.set_page_config(page_title="Bonds Dashboard", layout="wide")
st.title("📈 Bonds Dashboard")

# ===========================
# Custom CSS for blue dropdown text
# ===========================
st.markdown("""
<style>
/* Top dropdown font color */
div[data-baseweb="select"] > div {
    color: #1a73e8 !important;  /* blue text */
    font-size: 16px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# Top Dropdown Menu
# ===========================
section = st.selectbox(
    "Select subsection:",
    ["Zero-Coupon Bond", "Coupon-Paying Bond", "Analytics"]
)

# =======================
# Sidebar Inputs
# =======================
if section == "Zero-Coupon Bond":
    st.sidebar.header("Zero-Coupon Bond Inputs")
    F = st.sidebar.number_input("Face Value ($)", 100, 1_000_000, 1000)
    r_zcb = st.sidebar.slider("Yield (%)", 0.0, 20.0, 3.0) / 100
    T = st.sidebar.slider("Maturity (years)", 0.1, 30.0, 5.0)
    freq = st.sidebar.selectbox("Compounding Frequency", ["Annual", "Semi-Annual", "Quarterly", "Monthly"])
    

elif section == "Coupon-Paying Bond":
    st.sidebar.header("Coupon-Paying Bond Inputs")
    F = st.sidebar.number_input("Face Value ($)", 100, 1_000_000, 1000)
    C = st.sidebar.number_input("Annual Coupon ($)", 0, 100_000, 50)
    N = st.sidebar.slider("Number of Years (Maturity)", 1, 50, 5)
    r_coupon = st.sidebar.slider("Yield (%)", 0.0, 20.0, 3.0) / 100

elif section == "Analytics":
    st.sidebar.header("Analytics Inputs")
    F = st.sidebar.number_input("Face Value ($)", 100, 1_000_000, 1000, key="analytics_F")
    C = st.sidebar.number_input("Annual Coupon ($)", 0, 100_000, 50, key="analytics_C")
    N = st.sidebar.slider("Number of Years (Maturity)", 1, 50, 5, key="analytics_N")
    r_percent = st.sidebar.slider("Yield / YTM (%)", min_value=0.0, max_value=10.0, value=4.0, step=0.5, key="analytics_r_coupon")   
    r = r_percent / 100





# ===============================
# US Treasury bond data
# ===============================
@st.cache_data
def get_latest_yf_curve():
    symbols = {'3M':'^IRX', '5Y':'^FVX', '10Y':'^TNX', '30Y':'^TYX'}
    data = {}
    for maturity, ticker_symbol in symbols.items():
        ticker = yf.Ticker(ticker_symbol)
        # Get latest close price and convert to %
        data[maturity] = ticker.history(period="1d")['Close'][-1]
        
    df = pd.DataFrame({
        "Maturity": list(data.keys()),
        "Yield (%)": [round(y, 2) for y in data.values()]
    })

    # Optional: set Maturity as index to remove default integer index
    df.set_index("Maturity", inplace=True)
    return df

curve_df = get_latest_yf_curve()








# ===========================
# Zero-Coupon Bond Section
# ===========================
if section == "Zero-Coupon Bond":

    # ===============================
    # Header & Explanation
    # ===============================
    st.header("Zero-Coupon Bond (Vanilla Bond)")
    st.write(
        "A zero-coupon bond (Vanilla Bond) is a financial instrument (contract) between an issuer (borrower) and an investor (lender). "
        "The issuer borrows a specific amount of money from the investor, promising to pay back the face value"
        " at a predetermined maturity date. Unlike coupon-paying bonds, it does not make periodic "
        "interest payments. Instead, it is sold at a discount to its face value. The investor's return "
        "comes straight from the difference between the purchase price and the face value.\n\n"
        "The bond's price reflects the present value of the single future payment, discounted using the yield "
        "(interest rate) over the bond's duration.\n\n"
        "This highlights the **time value of money**: a dollar today is worth more than a dollar in the future. "
    )
    st.markdown("---")
    st.markdown("### 📘 Annual Compounding Formula")
    st.latex(r"Bond \ Price = \frac{FV}{(1+r)^T}")
    st.write(
        "This formula assumes interest compounds once per year. The yield \( r \) represents the annualized "
        "rate of return. The bond price decreases if the yield increases or the maturity lengthens."
    )


    # ===============================
    # Price under annual compounding
    # ===============================
    ZCB_price = F / (1 + r_zcb)**T
    T_values = np.linspace(0.1, 30, 300)
    zcb_prices_vs_T = F / (1 + r_zcb)**T_values


 

    # ===============================
    # Layout: Inputs + Graph
    # ===============================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Inputs:")
        st.write(f"**Face Value:** ${F:,.0f}")
        st.write(f"**Maturity:** {T} years")
        st.write(f"**Yield:** {r_zcb*100:.2f}%")
        st.write(f"#### Annual Compounding Bond Price: **${ZCB_price:,.2f}**\n\n")
        st.write("The US Government offers different types of zero-coupon bonds, known as Treasury Bills (T-Bills), with different maturities.\n"
                "Latest US Treasury Yields (via Yahoo Finance")
        st.table(curve_df)

    with col2:
        st.subheader("Price vs Maturity (Annual Compounding)")
        fig, ax = plt.subplots()
        ax.plot(T_values, zcb_prices_vs_T)
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Bond Price ($)")
        ax.set_title("Zero-Coupon Bond Price vs Maturity")
        st.pyplot(fig)

    st.markdown("---")

    # ==================================================
    # 🔄 Discrete Compounding Frequency
    # ==================================================
    st.header("📊 Discrete Compounding Frequency Comparison")
    st.write(
        "Interest does not always compound annually. In practice, it may compound multiple times per year, "
        "such as semi-annually, quarterly, or monthly. The more frequent the compounding, the higher the effective yield, "
        "and the bond price adjusts accordingly. This section allows you to select a compounding frequency and "
        "see its effect on the bond price."
    )


    freq_map = {"Annual": 1, "Semi-Annual": 2, "Quarterly": 4, "Monthly": 12}
    n = freq_map[freq]

    ZCB_price_disc = F / (1 + r_zcb/n)**(n*T)
    zcb_prices_disc_vs_T = F / (1 + r_zcb/n)**(n*T_values)

    d1, d2 = st.columns(2)
    with d1:
        st.subheader("Price Calculation (Discrete Compounding)")
        st.write(f"**Compounding Frequency:** {freq}")
        st.latex(r"Bond \ Price  = \frac{FV}{(1+\frac{r}{n})^{nT}}")
        st.subheader("Inputs:")
        st.write(f"**Face Value:** ${F:,.0f}")
        st.write(f"**Maturity:** {T} years")
        st.write(f"**Yield:** {r_zcb*100:.2f}%")
        st.write(f"#### Annual Bond Price: **${ZCB_price:,.2f}**")
        st.write(f"#### {freq} Bond Price: **${ZCB_price_disc:,.2f}**")

    with d2:
        st.subheader("Price vs Maturity")
        fig_disc, ax_disc = plt.subplots()
        ax_disc.plot(T_values, zcb_prices_disc_vs_T)
        ax_disc.set_xlabel("Maturity (years)")
        ax_disc.set_ylabel("Bond Price ($)")
        ax_disc.set_title(f"ZCB Price vs Maturity ({freq} Compounding)")
        st.pyplot(fig_disc)

    st.markdown("---")

    # ===============================
    # Continuous Compounding
    # ===============================
    st.header("📈 Continuous Compounding")
    st.write(
        "In the theoretical limit, if interest is compounded **continuously** (infinitely often), "
        "we use the formula for continuous discounting. This reflects a slightly lower present value "
        "because discounting occurs more frequently than with any discrete compounding interval."
    )
    st.latex(r"Bond \ Price = FV * e^{-r*T}")

    ZCB_price_cont = F * np.exp(-r_zcb * T)
    zcb_prices_cont_vs_T = F * np.exp(-r_zcb * T_values)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("Price Comparison")
        st.write(f"#### Annual Compounding Price: **${ZCB_price:,.2f}**")
        st.write(f"#### Continuous Compounding Price: **${ZCB_price_cont:,.2f}**")
        st.write(
            "Continuous compounding discounts more frequently, so the present value is slightly lower. "
            "The difference grows with longer maturities or higher yields. \n\n"
            "As maturity increases, the present value of a bond decreases because discounting has more time "
            "to reduce the future payout. The difference between annual and continuous compounding becomes relatively "
            "small over long horizons, so the two prices appear to converge. Essentially, both methods are dominated by "
            "exponential discounting, making their bond prices nearly identical for very long maturities."
        )

    with cc2:
        st.subheader("Annual vs Continuous Compounding")
        fig_cc, ax_cc = plt.subplots()
        ax_cc.plot(T_values, zcb_prices_cont_vs_T, label="Continuous")
        ax_cc.plot(T_values, zcb_prices_vs_T, linestyle="--", label="Annual")
        ax_cc.set_xlabel("Maturity (years)")
        ax_cc.set_ylabel("Bond Price ($)")
        ax_cc.set_title("Price vs Maturity: Annual vs Continuous")
        ax_cc.legend()
        st.pyplot(fig_cc)







# ===========================
# Coupon-Paying Bond Section
# ===========================

elif section == "Coupon-Paying Bond":


    # ===============================
    # Header & Explanation
    # ===============================
    st.header("Coupon-Paying Bond")
    st.write(
        "A coupon-paying bond is a financial instrument where the issuer borrows money from investors "
        "and promises to pay periodic interest (coupons) along with the face value at maturity. "
        "The price of the bond reflects the present value of all future cash flows, "
        "both the coupons and the final principal repayment.\n\n"
        "This illustrates the **time value of money**: each future cash flow is discounted back to today's value "
        "using the bond's yield (interest rate)."
    )
    st.markdown("---")

    # ===============================
    # Price Formula
    # ===============================
    st.markdown("### 📘 Price Formula (Annual Compounding)")
    st.latex(r"Price = \sum_{i=1}^{N} \frac{C}{(1+r)^i} + \frac{FV}{(1+r)^N}")
    st.write(
        "Where C is the coupon payment, FV is the face value, r is the annual yield, "
        "and N is the number of periods (years)."
    )

    # ===============================
    # Bond Price Calculations
    # ===============================
    cash_flows = [C] * N
    cash_flows[-1] += F  # add face value to last payment

    # Annual compounding
    coupon_price = sum(cf / (1 + r_coupon)**(i+1) for i, cf in enumerate(cash_flows))
    # Continuous compounding
    coupon_price_cont = sum(cf * np.exp(-r_coupon*(i+1)) for i, cf in enumerate(cash_flows))

    # Price vs maturity for chart
    T_values = np.linspace(1, N, 100)
    prices_ann = [
        sum(cf / (1 + r_coupon)**(i+1) for i, cf in enumerate(cash_flows[:int(t)] + [cash_flows[int(t)-1]])) 
        for t in T_values
    ]
    prices_cont = [
        sum(cf * np.exp(-r_coupon*(i+1)) for i, cf in enumerate(cash_flows[:int(t)] + [cash_flows[int(t)-1]])) 
        for t in T_values
    ]

    # ===============================
    # Current Yield and Macaulay Duration
    # ===============================
    current_yield = C / coupon_price



    # ===============================
    # Cash Flow Table
    # ===============================
    df = pd.DataFrame({
        "Year": range(1, N+1),
        "Coupon": [C]*N,
        "Total CF": cash_flows,
        "PV (annual)": [cf/(1+r_coupon)**(i+1) for i, cf in enumerate(cash_flows)],
    })

    # Add total PV row
    df = pd.concat([df, pd.DataFrame({
        "Year": ["Total PV"],
        "Coupon": [""],
        "Total CF": [""],
        "PV (annual)": [df["PV (annual)"].sum()]
    })], ignore_index=True)

    # Set 'Year' as the index to remove default integer index
    df.set_index("Year", inplace=True)

    # Display in Streamlit
   


    # ===============================
    # Layout
    # ===============================
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Inputs & Price")
        st.write(f"**Face Value:** ${F:,.0f}")
        st.write(f"**Annual Coupon:** ${C:,.0f}")
        st.write(f"**Maturity (Years):** {N}")
        st.write(f"**Yield:** {r_coupon*100:.2f}%")
        st.write(f"#### Price (Annual Compounding): **${coupon_price:,.2f}**")
        st.write(f"#### Price (Continuous Compounding): **${coupon_price_cont:,.2f}**")
        st.write(f"#### Current Yield: **{current_yield*100:.2f}%**")
        

    with col2:
        st.subheader("Price vs Maturity")
        fig, ax = plt.subplots()
        ax.plot(T_values, prices_ann, linestyle="--", label="Annual Compounding")
        ax.plot(T_values, prices_cont, label="Continuous Compounding")
        ax.set_xlabel("Years")
        ax.set_ylabel("Bond Price ($)")
        ax.set_title("Coupon-Paying Bond: Annual vs Continuous Compounding")
        ax.legend()
        st.pyplot(fig)

    st.write ("---")

    st.write("💡 **Yield Used for Discounting Future Cash Flows:**\n"
        "- The bond price is calculated by discounting **all future cash flows** (coupons + principal) using the **input yield / Yield to Maturity (YTM)**.\n"
        "- **Current yield** (annual coupon divided by bond price) is only a measure of **income relative to price**, and does **not account for principal repayment** or timing of cash flows.\n"
        "- **Intuition:** YTM represents the total return an investor earns if the bond is held to maturity, which is why it is used for discounting.")
    st.write ("---")
    st.subheader("Cash Flow Table")
    st.table(df)
    st.write ("---")
    st.write(
        "### 🔹 Staircase Pattern Explanation\n"
        "The small 'steps' seen in the bond price vs maturity graph occur because the bond has **discrete annual coupon payments**. "
        "Each step represents the effect of a new coupon payment being added to the present value calculation. "
        "As each coupon payment occurs at the end of a year, the price increases discretely rather than smoothly, "
        "creating the staircase appearance. Continuous compounding smooths out these steps, resulting in a smoother price curve."
    )










# ===========================
# Analytics Section
# ===========================
elif section == "Analytics":

    st.header("Bond Analytics – Duration, Convexity & Price-Yield Analysis")
    st.write(
        "This section explores key bond metrics that help investors understand **price sensitivity, "
        "interest rate risk, and overall bond behavior**. "
        "We cover duration, convexity, and price-yield relationships, with formulas, intuition, examples, and graphical illustrations."
    )
    st.markdown("---")


    # ===============================
    # Duration Section
    # ===============================
    st.subheader("📏 Duration: Measuring Interest Rate Sensitivity")
    st.write(
        "Duration is a key concept in bond investing. It tells us **how sensitive a bond's price is to changes in interest rates** "
        "and can also be interpreted as the **average time it takes to receive all the bond's cash flows**, weighted by their size. "
        "Understanding duration helps investors manage risk and compare bonds on a common scale, even if they have different maturities or coupons."
    )

    st.markdown(
        "💡 **Intuition:**\n"
        "- Think of duration as a **risk meter for bonds**: the higher the duration, the more a bond's price will fluctuate if interest rates change.\n"
        "- **Zero-coupon bonds:** Duration equals maturity because all cash comes at the end.\n"
        "- **Coupon-paying bonds:** Duration is shorter than maturity because you receive some cash earlier.\n"
        "- Duration depends on three main factors:\n"
        "  1. **Time to maturity** – longer maturities generally increase duration.\n"
        "  2. **Coupon size** – higher coupons shorten duration.\n"
        "  3. **Price vs interest rate environment** – premium or discount bonds slightly alter duration.\n"
        "- Example: A 10-year bond with 50 USD annual coupon and 1000 USD face value has an 'average waiting time' of ~7–8 years. This indicates how sensitive its price is to interest rate changes."
    )

    # ===============================
    # Formulas
    # ===============================

    cash_flows = [C]*N
    cash_flows[-1] += F
    price = sum(cf / (1+r)**(i+1) for i, cf in enumerate(cash_flows))


    st.latex(r"Macaulay\ Duration:\ D_{Mac} = \frac{\sum_{i=1}^{N} t_i \cdot PV(CF_i)}{Price}")
    st.latex(r"Modified\ Duration:\ D_{Mod} = \frac{D_{Mac}}{1+r}")

    # Calculate Macaulay and Modified Duration
    macaulay_duration = sum((i+1) * cf / (1+r)**(i+1) for i, cf in enumerate(cash_flows)) / price
    modified_duration = macaulay_duration / (1+r)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"#### Macaulay Duration: {macaulay_duration:.2f} years")
        st.write(f"#### Modified Duration: {modified_duration:.2f} years")
        st.markdown(
            "💡 **Takeaways:**\n"
            "- **Macaulay duration** shows the weighted average time to receive cash flows.\n"
            "- **Modified duration** estimates the **percentage change in bond price** for a 1% change in yield.\n"
            "- Longer duration → higher price sensitivity.\n"
            "- Mathematically, a bond with a modified duration of 5 years will see its price change by approximately 5% for a 1% change in yield."
        )

    # ===============================
    # Price Sensitivity Visualization
    # ===============================
    with col2:
        delta_yields = np.linspace(-0.05, 0.05, 100)
        prices_approx = price * (1 - modified_duration * delta_yields)
        prices_exact = [sum(cf / (1+(r+dy))**(i+1) for i, cf in enumerate(cash_flows)) for dy in delta_yields]

        fig, ax = plt.subplots()
        ax.plot(delta_yields*100, prices_exact, label="Exact Price")
        ax.plot(delta_yields*100, prices_approx, linestyle="--", label="Duration Approx.")
        ax.set_xlabel("Yield Change (%)")
        ax.set_ylabel("Bond Price ($)")
        ax.set_title("Price Sensitivity vs Yield Change")
        ax.legend()
        st.pyplot(fig)


    # ===============================
    # Duration vs Maturity Visualization (Approximation)
    # ===============================
    st.subheader("⏳ Duration vs Maturity")
    st.write(
        "We can visualize how duration changes with bond maturity. "
        "Longer maturities generally have longer durations, which means they are more sensitive to interest rate changes. "
        "This simple approximation assumes constant coupon payments and ignores the yield curve."
    )

    # Define maturity range
    maturity_range = np.arange(1, 31)  # 1 to 30 years

    # Compute approximate durations
    durations = []
    for T_m in maturity_range:
        cash_flows_m = [C]*T_m
        cash_flows_m[-1] += F
        duration = sum((i+1)*cf for i, cf in enumerate(cash_flows_m)) / sum(cash_flows_m)
        durations.append(duration)

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "💡 **Takeaways:**\n"
            "- Duration generally **increases with maturity** for bonds with fixed coupons.\n"
            "- Bonds with **longer maturities** are more sensitive to interest rate changes.\n"
            "- **Zero-coupon bonds** have durations equal to their maturity.\n"
            "- This is because no interest is being paid before maturity, so all cash flows occur at the end."
        )

    with col2:
        fig, ax = plt.subplots()
        ax.plot(maturity_range, durations, marker='o')
        ax.set_xlabel("Maturity (Years)")
        ax.set_ylabel("Duration (Years)")
        ax.set_title("Duration vs Maturity (Approximation)")
        st.pyplot(fig)

    st.write("Hence, duration provides a useful measure of interest rate risk (change sensitivity) and average time to cash flows for bonds.\n\n"
             "At the end of the page we will cover one concrete example illustrating price changes with interest rate shifts."
                )
    st.markdown("---")



   # ===============================
    # Yield-to-Maturity vs Maturity Curve
    # ===============================
    st.subheader("💹 Yield-to-Maturity")
    st.write(
        "The **Yield-to-Maturity (YTM)** is the discount rate that makes the present value of a bond's cash flows equal to its current price. "
        "This plot shows how the YTM changes for different maturities, keeping the bond's price fixed."
    )
    st.latex(r"Bond \ Price = \sum_{i=1}^{N} \frac{CF_i}{(1+YTM)^{t_i}}")

    st.markdown(
        "💡 **Intuition:**\n"
        "- Longer maturities usually have higher yields for a given price due to **increased interest rate risk**.\n"
        "- This curve helps investors understand the relationship between **time to maturity** and required yield.\n"
        "- Example: A bond priced at 950 USD with 50 USD annual coupon may have a YTM of 4.5% for 5 years, but 5% for 10 years."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "- The curve illustrates how **yield adjusts with maturity** for a fixed bond price.\n"
            "- Investors can see the trade-off between longer maturities and higher required yields.\n"
            "- Useful for **pricing bonds with different maturities** or building a **yield curve**."
        )
        st.write("Latest US Treasury Yields (via Yahoo Finance)")
        st.table(curve_df)

        st.write("Note that the 3M yield is annualized T-bill yield, not a coupon-bearing bond.")



    # --- Plot YTM and Treasury curve ---
    with col2:
        fig, ax = plt.subplots(figsize=(8,5))
        
        # Map maturities to numeric years
        maturity_map = {"3M": 0.25, "5Y": 5, "10Y": 10, "30Y": 30}
        treasury_maturities = [maturity_map[m] for m in curve_df.index]  # use index now
        treasury_yields = curve_df["Yield (%)"].values  # already in %
        
        # Plot the curve
        ax.plot(treasury_maturities, treasury_yields, 
                label="US Treasury Yield Curve", marker='o', linestyle="--", color="orange")
        
        # Labels & title
        ax.set_xlabel("Maturity (Years)")
        ax.set_ylabel("Treasury Yield (%)")
        ax.set_title("Treasury Yield Curve")
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)





    st.markdown("---")

    # ===============================
    # Price Change Example with Interest Rate Shift
    # ===============================
    st.subheader("💡 Price Change Example (Interest Rate Shift)")

    st.write(
        "Let's illustrate how a change in interest rates affects the bond price using **Modified Duration** "
        "for a concrete example."
    )

    # Use the same bond inputs as above (F, C, N, r_coupon)
    F_ex = F
    C_ex = C
    N_ex = N
    r_ex = r

    # Interest rate shift slider (new input)
 
    delta_y_percent = st.sidebar.slider("Interest Rate Change (%)", min_value=-5.0, max_value=5.0, value=2.0, step=0.5)
    delta_y_ex = delta_y_percent / 100


    # Calculate cash flows and price
    cash_flows_ex = [C_ex]*N_ex
    cash_flows_ex[-1] += F_ex
    price_ex = sum(cf / (1+r_ex)**(i+1) for i, cf in enumerate(cash_flows_ex))

    # Calculate Macaulay and Modified Duration
    macaulay_ex = sum((i+1)*cf / (1+r_ex)**(i+1) for i, cf in enumerate(cash_flows_ex)) / price_ex
    modified_ex = macaulay_ex / (1+r_ex)

    # Price change using duration approximation
        # Calculate approximate and exact price changes
    price_change_duration = - modified_ex * delta_y_ex * price_ex
    price_exact_ex = sum(cf / (1+r_ex+delta_y_ex)**(i+1) for i, cf in enumerate(cash_flows_ex))

    # Display metrics with explanation
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Bond Example:** {N_ex}-year bond, Face Value {F_ex} USD, Annual Coupon {C_ex} USD, Yield {r_ex*100:.2f}%")
        
        # Show original price
        st.write(f"1️⃣ Current Price (using YTM): **${price_ex:,.2f}**")
    
        # Show modified duration
        st.write(f"\n2️⃣ Modified Duration: **{modified_ex:.2f} years**")

        # Show interest rate change
        st.write(f"\n3️⃣ Interest Rate Change: **{delta_y_ex*100:.2f}%**")

        # Approximate price change using duration
        st.write(f"4️⃣ Approximate Price Change (Duration):")
        st.write(f"   ΔPrice ≈ - Duration × ΔYield × Current Price")
        st.write(f"   ΔPrice ≈ - {modified_ex:.2f} × {delta_y_ex*100:.2f}% × USD {price_ex:,.2f} = USD {price_change_duration:,.2f}")
        st.write(f"   → New Price (Approx.) = **${price_ex + price_change_duration:,.2f}**")

        # Exact price after yield change
        st.write(f"\n5️⃣ Exact Price after yield change:")
        st.write(f"   - Calculated by discounting each cash flow using the new yield: {(r_ex + delta_y_ex)*100:.2f}%")
        st.write(f"   → Exact Price = **${price_exact_ex:,.2f}**")

        st.markdown(
            "💡 **Intuition:**\n"
            "- Duration provides a linear estimate of price change for small yield changes.\n"
            "- Exact calculation shows the actual price impact.\n"
            "- This illustrates why longer-term, low-coupon bonds are more sensitive to interest rate shifts."
    )


    # Price vs yield change plot
    with col2:
        delta_y_ex_vals = np.linspace(-0.05, 0.05, 100)
        prices_exact_plot = [sum(cf / (1+r_ex+dy)**(i+1) for i, cf in enumerate(cash_flows_ex)) for dy in delta_y_ex_vals]
        prices_duration_plot = [price_ex*(1 - modified_ex*dy) for dy in delta_y_ex_vals]

        fig, ax = plt.subplots()
        ax.plot(delta_y_ex_vals*100, prices_exact_plot, label="Exact Price")
        ax.plot(delta_y_ex_vals*100, prices_duration_plot, linestyle="--", label="Duration Approx.")
        ax.set_xlabel("Yield Change (%)")
        ax.set_ylabel("Bond Price ($)")
        ax.set_title("Price Change vs Yield Shift")
        ax.legend()
        st.pyplot(fig)
