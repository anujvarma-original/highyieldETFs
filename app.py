from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analytics import (
    analyze_ticker,
    download_history,
    price_series,
    project_income,
    random_portfolio_search,
    total_return_index,
)
from etf_catalog import DEFAULT_TICKERS, ETF_DATABASE

st.set_page_config(page_title="High-Income ETF Research Platform", page_icon="💵", layout="wide")

@st.cache_data(ttl=3600, show_spinner=False)
def benchmark_returns(ticker: str, period: str) -> pd.Series:
    history = download_history(ticker, period)
    return total_return_index(history).pct_change().dropna()

def fmt_money(value):
    return "N/A" if pd.isna(value) else f"${value:,.2f}"

st.title("High-Income ETF Research Platform")
st.caption("Covered-call, 0DTE, volatility-premium and crypto-option income ETFs. Historical distributions are not guaranteed future income.")

with st.sidebar:
    st.header("Analysis Settings")
    selected_tickers = st.multiselect("ETFs", DEFAULT_TICKERS, default=DEFAULT_TICKERS[:17])
    custom = st.text_input("Additional tickers", placeholder="Example: XQQI, CEPI")
    if custom:
        for ticker in [x.strip().upper() for x in custom.replace(";", ",").split(",") if x.strip()]:
            if ticker not in selected_tickers:
                selected_tickers.append(ticker)
    investment = st.number_input("Investment in each ETF", 1000.0, 10_000_000.0, 100_000.0, 10_000.0)
    period = st.selectbox("History", ["1y", "2y", "5y", "10y", "max"], index=2)
    risk_free_rate = st.number_input("Risk-free rate", 0.0, 0.20, 0.04, 0.005, format="%.3f")
    run = st.button("Analyze ETFs", type="primary", use_container_width=True)

catalog_rows = [{"Ticker": ticker, **meta} for ticker, meta in ETF_DATABASE.items()]
catalog_df = pd.DataFrame(catalog_rows).rename(columns={
    "name": "ETF Name", "issuer": "Issuer", "strategy": "Strategy", "exposure": "Exposure",
    "frequency": "Distribution Frequency", "expense_ratio": "Expense Ratio",
    "risk_band": "Risk Band", "roc_note": "Return-of-Capital Note"
})

tab_overview, tab_lookup, tab_income, tab_compare, tab_portfolio, tab_projection, tab_method = st.tabs([
    "Overview", "ETF Lookup", "Income History", "Compare ETFs", "Portfolio Builder", "Income Projection", "Methodology"
])

with tab_lookup:
    st.subheader("ETF Strategy Lookup")
    query = st.text_input("Search lookup table", placeholder="Issuer, index, strategy or ticker")
    lookup = catalog_df.copy()
    if query:
        mask = lookup.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        lookup = lookup[mask]
    st.dataframe(
        lookup,
        use_container_width=True,
        hide_index=True,
        column_config={"Expense Ratio": st.column_config.NumberColumn(format="%.2f%%")},
    )
    st.caption("Expense ratios and distribution schedules are maintained as reference metadata and should be checked against current prospectuses before investing.")

if not selected_tickers:
    with tab_overview:
        st.warning("Select at least one ticker.")
    st.stop()

spy_returns = benchmark_returns("SPY", period)
qqq_returns = benchmark_returns("QQQ", period)

rows, monthly_frames, total_return_frames, errors = [], [], [], []
progress = st.progress(0)
for i, ticker in enumerate(selected_tickers):
    metadata = ETF_DATABASE.get(ticker, {
        "name": ticker, "issuer": "Custom", "strategy": "Unknown", "exposure": "Unknown",
        "frequency": "Unknown", "expense_ratio": np.nan, "risk_band": "Unclassified",
        "roc_note": "No catalog entry; verify prospectus and tax documents."
    })
    row, monthly, total_frame, error = analyze_ticker(
        ticker, metadata, investment, period, spy_returns, qqq_returns, risk_free_rate
    )
    if row:
        rows.append(row)
    if not monthly.empty:
        monthly_frames.append(monthly)
    if not total_frame.empty:
        total_return_frames.append(total_frame)
    if error:
        errors.append(error)
    progress.progress((i + 1) / len(selected_tickers))
progress.empty()

if not rows:
    with tab_overview:
        st.error("No ETF data was returned. Review the ticker symbols and Yahoo Finance availability.")
        for error in errors:
            st.warning(error)
    st.stop()

results = pd.DataFrame(rows)
monthly_all = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
total_all = pd.concat(total_return_frames, ignore_index=True) if total_return_frames else pd.DataFrame()

with tab_overview:
    total_investment = results["Investment"].sum()
    annual_income = results["Estimated Annual Income"].sum()
    monthly_income = results["Estimated Monthly Income"].sum()
    portfolio_yield = annual_income / total_investment if total_investment else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Modeled Investment", f"${total_investment:,.0f}")
    c2.metric("Estimated Monthly Income", fmt_money(monthly_income))
    c3.metric("Estimated Annual Income", fmt_money(annual_income))
    c4.metric("Average Portfolio Yield", f"{portfolio_yield:.2%}")

    columns = [
        "Ticker", "ETF Name", "Strategy", "Exposure", "Distribution Frequency", "Risk Band",
        "Current Price", "Shares", "TTM Distribution Yield", "Estimated Monthly Income",
        "Estimated Annual Income", "Forecast Position Payment", "Forecast Confidence",
        "Income Stability Score", "Total Return CAGR", "Maximum Drawdown", "Annualized Volatility",
        "Sharpe Ratio", "Sortino Ratio", "Ulcer Index", "Beta vs SPY", "Correlation vs SPY",
        "Correlation vs QQQ", "Expense Ratio"
    ]
    st.dataframe(
        results[columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Current Price": st.column_config.NumberColumn(format="$%.2f"),
            "Shares": st.column_config.NumberColumn(format="%.2f"),
            "TTM Distribution Yield": st.column_config.NumberColumn(format="%.2f%%"),
            "Estimated Monthly Income": st.column_config.NumberColumn(format="$%.2f"),
            "Estimated Annual Income": st.column_config.NumberColumn(format="$%.2f"),
            "Forecast Position Payment": st.column_config.NumberColumn(format="$%.2f"),
            "Total Return CAGR": st.column_config.NumberColumn(format="%.2f%%"),
            "Maximum Drawdown": st.column_config.NumberColumn(format="%.2f%%"),
            "Annualized Volatility": st.column_config.NumberColumn(format="%.2f%%"),
            "Expense Ratio": st.column_config.NumberColumn(format="%.2f%%"),
            "Income Stability Score": st.column_config.ProgressColumn(min_value=0, max_value=100),
        },
    )

    chart = px.bar(results.sort_values("Estimated Monthly Income", ascending=False), x="Ticker", y="Estimated Monthly Income",
                   title=f"Estimated Average Monthly Income on ${investment:,.0f} per ETF")
    chart.update_layout(yaxis_tickprefix="$")
    st.plotly_chart(chart, use_container_width=True)

    if not total_all.empty:
        performance = px.line(total_all, x="Date", y="Total Return Index", color="Ticker",
                              title="Growth of $1 with Distributions Reinvested")
        st.plotly_chart(performance, use_container_width=True)

    if errors:
        with st.expander(f"Data warnings ({len(errors)})"):
            for error in errors:
                st.warning(error)

with tab_income:
    st.subheader("Historical Distribution Income")
    if monthly_all.empty:
        st.info("No distribution history was returned.")
    else:
        chosen = st.multiselect("Tickers to chart", results["Ticker"].tolist(), default=results["Ticker"].tolist()[:6], key="income_chart")
        months = st.slider("Months shown", 6, 60, 24)
        cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=months)
        filtered = monthly_all[(monthly_all["Ticker"].isin(chosen)) & (monthly_all["Month"] >= cutoff)]
        fig = px.line(filtered, x="Month", y="Position Income", color="Ticker", markers=True,
                      title=f"Actual Historical Cash Distributions for a ${investment:,.0f} Modeled Position")
        fig.update_layout(yaxis_tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(filtered.sort_values(["Month", "Ticker"], ascending=[False, True]), use_container_width=True, hide_index=True)

with tab_compare:
    st.subheader("Side-by-Side ETF Comparison")
    options = results["Ticker"].tolist()
    first = st.selectbox("ETF 1", options, index=0)
    second = st.selectbox("ETF 2", options, index=min(1, len(options)-1))
    compare_metrics = [
        "TTM Distribution Yield", "Estimated Monthly Income", "Income Stability Score",
        "Total Return CAGR", "Maximum Drawdown", "Annualized Volatility", "Sharpe Ratio",
        "Sortino Ratio", "Ulcer Index", "Expense Ratio", "Correlation vs SPY"
    ]
    comparison = results.set_index("Ticker").loc[[first, second], compare_metrics].T.reset_index()
    comparison.columns = ["Metric", first, second]
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.write("**Return-of-capital notes**")
    for ticker in [first, second]:
        note = results.loc[results["Ticker"] == ticker, "ROC Note"].iloc[0]
        st.write(f"**{ticker}:** {note}")

    if not total_all.empty:
        pair = total_all[total_all["Ticker"].isin([first, second])]
        st.plotly_chart(px.line(pair, x="Date", y="Total Return Index", color="Ticker",
                               title="Reinvested Total-Return Comparison"), use_container_width=True)

with tab_portfolio:
    st.subheader("Income Portfolio Builder")
    builder_tickers = st.multiselect("Portfolio candidates", results["Ticker"].tolist(),
                                     default=results.sort_values("Income Stability Score", ascending=False)["Ticker"].head(min(8, len(results))).tolist())
    objective = st.selectbox("Optimization objective", ["Balanced", "Maximum income", "Best Sharpe", "Lowest volatility"])
    simulations = st.slider("Random portfolios", 1000, 20000, 5000, step=1000)
    max_weight = st.slider("Maximum position weight", 0.10, 1.0, 0.35, 0.05)

    if len(builder_tickers) >= 2:
        return_series = {}
        for ticker in builder_tickers:
            hist = download_history(ticker, period)
            index = total_return_index(hist)
            return_series[ticker] = index.pct_change()
        daily_returns = pd.DataFrame(return_series).dropna(how="all")
        income_per_dollar = results.set_index("Ticker")["Estimated Annual Income"] / results.set_index("Ticker")["Investment"]
        optimized = random_portfolio_search(
            daily_returns, results.set_index("Ticker")["TTM Distribution Yield"],
            income_per_dollar, simulations, max_weight, objective, risk_free_rate
        )
        if optimized.empty:
            st.warning("No portfolios satisfied the weight constraint. Increase the maximum position weight.")
        else:
            best = optimized.iloc[0]
            weights = pd.DataFrame({
                "Ticker": builder_tickers,
                "Weight": [best.get(f"Weight {ticker}", 0) for ticker in builder_tickers],
            }).sort_values("Weight", ascending=False)
            total_capital = st.number_input("Portfolio capital", 10_000.0, 100_000_000.0, 400_000.0, 10_000.0)
            weights["Allocation"] = weights["Weight"] * total_capital
            weights["Estimated Annual Income"] = weights["Ticker"].map(income_per_dollar) * weights["Allocation"]
            weights["Estimated Monthly Income"] = weights["Estimated Annual Income"] / 12
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Portfolio Yield", f"{best['Income Yield']:.2%}")
            c2.metric("Annual Income", fmt_money(weights["Estimated Annual Income"].sum()))
            c3.metric("Monthly Income", fmt_money(weights["Estimated Monthly Income"].sum()))
            c4.metric("Modeled Volatility", f"{best['Volatility']:.2%}")
            st.dataframe(weights, use_container_width=True, hide_index=True,
                         column_config={"Weight": st.column_config.NumberColumn(format="%.2f%%"),
                                        "Allocation": st.column_config.NumberColumn(format="$%.0f"),
                                        "Estimated Annual Income": st.column_config.NumberColumn(format="$%.2f"),
                                        "Estimated Monthly Income": st.column_config.NumberColumn(format="$%.2f")})
            st.plotly_chart(px.pie(weights, names="Ticker", values="Weight", title="Suggested Allocation"), use_container_width=True)

            corr = daily_returns.corr()
            st.subheader("Correlation Matrix")
            st.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto"), use_container_width=True)
    else:
        st.info("Choose at least two ETFs.")

with tab_projection:
    st.subheader("Long-Term Income Projection")
    starting = st.number_input("Starting portfolio", 1_000.0, 100_000_000.0, 400_000.0, 10_000.0, key="projection_start")
    contribution = st.number_input("Monthly contribution", 0.0, 1_000_000.0, 2_000.0, 500.0)
    years = st.slider("Projection years", 1, 40, 20)
    reinvest = st.checkbox("Reinvest distributions", True)
    assumed_yield = st.number_input("Assumed annual distribution yield", 0.0, 1.0, float(portfolio_yield), 0.005, format="%.3f")
    default_return = float(results["Total Return CAGR"].replace([np.inf, -np.inf], np.nan).median())
    if pd.isna(default_return):
        default_return = assumed_yield
    assumed_return = st.number_input("Assumed annual total return", -0.50, 1.0, default_return, 0.01, format="%.3f")
    projection = project_income(starting, assumed_yield, assumed_return, contribution, years, reinvest)
    st.plotly_chart(px.line(projection, x="Year", y="Portfolio Value", title="Projected Portfolio Value"), use_container_width=True)
    st.plotly_chart(px.line(projection, x="Year", y="Monthly Income at Current Yield", title="Projected Monthly Income"), use_container_width=True)
    st.dataframe(projection, use_container_width=True, hide_index=True,
                 column_config={"Portfolio Value": st.column_config.NumberColumn(format="$%.0f"),
                                "Annual Income at Current Yield": st.column_config.NumberColumn(format="$%.0f"),
                                "Monthly Income at Current Yield": st.column_config.NumberColumn(format="$%.0f"),
                                "Cumulative Contributions": st.column_config.NumberColumn(format="$%.0f"),
                                "Yield on Original Capital": st.column_config.NumberColumn(format="%.2f%%")})

with tab_method:
    st.subheader("Methodology and Important Limitations")
    st.markdown("""
- **Income:** trailing 365-day cash distributions multiplied by shares purchased at the latest price.
- **Total return:** price changes plus distributions, assuming distributions are reinvested.
- **Stability score:** payout variability, cuts, skipped monthly payments and recent payout growth. New funds naturally receive less-reliable scores.
- **Forecast:** recency-weighted average of recent per-share payments. It is a historical estimate, not an options-pricing forecast.
- **Optimizer:** random portfolio search using historical total-return observations. It does not guarantee a globally optimal portfolio.
- **Return of capital:** Yahoo Finance does not reliably identify the tax character of every payment. The app displays a warning note, but definitive ROC data must come from the fund's Section 19 notices and annual tax documents.
- **Expense ratios and schedules:** catalog values are reference values and can change. Verify them against the current prospectus.
- **Yield is not total return:** very high distributions may coexist with NAV erosion, capped upside, higher volatility or return of capital.
""")

st.divider()
st.caption("Educational research only—not investment, tax, legal or accounting advice. Market and distribution data may be delayed, incomplete or revised.")
