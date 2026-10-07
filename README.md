# Derivatives Dashboard

An interactive Streamlit learning app for exploring bonds, futures, options, and swaps. It combines short explanations with user-controlled pricing examples, payoff charts, and risk measures.

**Live app:** https://hendrikesser-finance-dashboard.streamlit.app/

## Modules

- **Bonds:** zero-coupon and coupon bond pricing, compounding, yields, and duration.
- **Futures:** linear payoffs, hedging, cost of carry, FX parity, interest-rate futures, and VIX futures.
- **Options:** payoff structures, put-call parity, binomial pricing, Black–Scholes, and selected Greeks.
- **Swaps:** introductory FX, interest-rate, and credit default swap examples.

## Run locally

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run 00_Home.py
```

Some pages retrieve reference market data through Yahoo Finance using `yfinance`; those panels require an internet connection and may be unavailable when the data source is down.

## Model assumptions and limitations

This is an educational project, not a trading, investment, or valuation system. The examples use simplified assumptions such as flat rates, continuous compounding, annual coupon schedules, constant volatility or hazard rates, and frictionless markets. Actual instruments can use different day-count conventions, calendars, settlement rules, collateral terms, curves, and market conventions. The Eurodollar futures example is historical; CME SOFR futures are the current USD benchmark contracts. The multi-payment FX example is a stylized cross-currency swap illustration.

Model outputs and educational explanations can contain errors. Check formulas and conventions against authoritative references before using them for coursework or professional decisions. Market data is provided by Yahoo Finance and is not guaranteed to be complete, timely, or suitable for valuation.

## Dependencies

Python packages and versions are listed in [`requirements.txt`](requirements.txt).
