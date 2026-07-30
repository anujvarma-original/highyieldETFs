"""
High-Yield ETF Income Analyzer
==============================

Streamlit application that analyzes high-yield and covered-call ETFs.

Main assumptions:
- $100,000 is invested in EACH ETF.
- Shares may be fractional.
- Trailing 12-month distributions are used to estimate annual income.
- Estimated monthly income equals trailing annual income divided by 12.
- Distribution amounts and yields are not guaranteed.
- Taxes, transaction costs, and distribution changes are excluded.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ---------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="High-Yield ETF Income Analyzer",
    page_icon="💵",
    layout="wide",
)


# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------

DEFAULT_TICKERS = [
    "QQQI",
    "JEPQ",
    "SPYI",
    "JEPI",
    "QYLD",
    "XYLD",
    "RYLD",
]

DEFAULT_INVESTMENT_PER_ETF = 100_000.0
TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------
# ETF DESCRIPTIONS
# ---------------------------------------------------------------------

ETF_NAMES = {
    "QQQI": "NEOS Nasdaq-100 High Income ETF",
    "JEPQ": "JPMorgan Nasdaq Equity Premium Income ETF",
    "SPYI": "NEOS S&P 500 High Income ETF",
    "JEPI": "JPMorgan Equity Premium Income ETF",
    "QYLD": "Global X Nasdaq 100 Covered Call ETF",
    "XYLD": "Global X S&P 500 Covered Call ETF",
    "RYLD": "Global X Russell 2000 Covered Call ETF",
    "DIVO": "Amplify CWP Enhanced Dividend Income ETF",
    "GPIX": "Goldman Sachs S&P 500 Premium Income ETF",
    "GPIQ": "Goldman Sachs Nasdaq-100 Premium Income ETF",
    "FEPI": "REX FANG & Innovation Equity Premium Income ETF",
    "AIPI": "REX AI Equity Premium Income ETF",
    "IWMI": "NEOS Russell 2000 High Income ETF",
    "WDTE": "Defiance S&P 500 Enhanced Options Income ETF",
    "XDTE": "Roundhill S&P 500 0DTE Covered Call Strategy ETF",
    "QDTE": "Roundhill Innovation-100 0DTE Covered Call Strategy ETF",
    "RDTE": "Roundhill Small Cap 0DTE Covered Call Strategy ETF",
    "SVOL": "Simplify Volatility Premium ETF",
}


# ---------------------------------------------------------------------
# FORMATTING HELPERS
# ---------------------------------------------------------------------

def format_currency(value: float | int | None) -> str:
    """Format a number as US currency."""

    if value is None or pd.isna(value):
        return "N/A"

    return f"${value:,.2f}"


def format_large_currency(value: float | int | None) -> str:
    """Format a number as currency without cents."""

    if value is None or pd.isna(value):
        return "N/A"

    return f"${value:,.0f}"


def format_percentage(value: float | int | None) -> str:
    """Format a decimal value as a percentage."""

    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.2%}"


def safe_float(value: Any) -> float:
    """Convert a value to float, returning NaN on failure."""

    try:
        converted = float(value)

        if np.isfinite(converted):
            return converted

        return np.nan

    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------
# TICKER INPUT
# ---------------------------------------------------------------------

def parse_tickers(raw_text: str) -> list[str]:
    """
    Convert comma-separated ticker input into a clean unique list.
    """

    cleaned_text = raw_text.replace("\n", ",").replace(";", ",")

    tickers: list[str] = []

    for item in cleaned_text.split(","):
        ticker = item.strip().upper()

        if ticker and ticker not in tickers:
            tickers.append(ticker)

    return tickers


# ---------------------------------------------------------------------
# MARKET DATA
# ---------------------------------------------------------------------

@st.cache_data(ttl=3_600, show_spinner=False)
def download_ticker_history(
    ticker_symbol: str,
    period: str = "5y",
) -> pd.DataFrame:
    """
    Download price and distribution history for one ticker.

    The returned history normally includes:
    - Open
    - High
    - Low
    - Close
    - Volume
    - Dividends
    - Stock Splits
    """

    ticker = yf.Ticker(ticker_symbol)

    history = ticker.history(
        period=period,
        auto_adjust=False,
        actions=True,
    )

    if history is None or history.empty:
        return pd.DataFrame()

    history = history.copy()

    # Remove timezone information to simplify date comparisons.
    if isinstance(history.index, pd.DatetimeIndex):
        try:
            history.index = history.index.tz_localize(None)
        except TypeError:
            pass

    history.sort_index(inplace=True)

    return history


@st.cache_data(ttl=3_600, show_spinner=False)
def download_ticker_metadata(ticker_symbol: str) -> dict[str, Any]:
    """
    Download optional ticker metadata.

    Yahoo metadata may be incomplete, so every use of this function
    must include a fallback.
    """

    try:
        ticker = yf.Ticker(ticker_symbol)
        metadata = ticker.info

        if isinstance(metadata, dict):
            return metadata

    except Exception:
        pass

    return {}


# ---------------------------------------------------------------------
# ANALYTICS
# ---------------------------------------------------------------------

def choose_price_column(history: pd.DataFrame) -> str | None:
    """Return the best available price column."""

    if "Close" in history.columns:
        return "Close"

    if "Adj Close" in history.columns:
        return "Adj Close"

    return None


def calculate_max_drawdown(price_series: pd.Series) -> float:
    """
    Calculate maximum drawdown from a price series.

    Returns a decimal:
        -0.25 means a 25% maximum drawdown.
    """

    prices = pd.to_numeric(price_series, errors="coerce").dropna()

    if prices.empty:
        return np.nan

    running_high = prices.cummax()
    drawdown = prices / running_high - 1.0

    return safe_float(drawdown.min())


def calculate_annualized_volatility(price_series: pd.Series) -> float:
    """
    Calculate annualized volatility from daily price returns.
    """

    prices = pd.to_numeric(price_series, errors="coerce").dropna()

    if len(prices) < 3:
        return np.nan

    daily_returns = prices.pct_change().dropna()

    if daily_returns.empty:
        return np.nan

    return safe_float(
        daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def calculate_price_return(
    price_series: pd.Series,
    start_date: pd.Timestamp,
) -> float:
    """
    Calculate price return from the first available price on or after
    start_date through the latest price.
    """

    prices = pd.to_numeric(price_series, errors="coerce").dropna()

    if prices.empty:
        return np.nan

    period_prices = prices.loc[prices.index >= start_date]

    if len(period_prices) < 2:
        return np.nan

    starting_price = safe_float(period_prices.iloc[0])
    ending_price = safe_float(period_prices.iloc[-1])

    if not np.isfinite(starting_price) or starting_price <= 0:
        return np.nan

    return ending_price / starting_price - 1.0


def calculate_distribution_growth(
    dividends: pd.Series,
) -> float:
    """
    Compare distributions during the latest 12 months with distributions
    during the preceding 12 months.

    Returns a decimal growth rate.
    """

    if dividends.empty:
        return np.nan

    latest_date = dividends.index.max()
    current_start = latest_date - pd.Timedelta(days=365)
    previous_start = current_start - pd.Timedelta(days=365)

    current_total = dividends.loc[
        dividends.index > current_start
    ].sum()

    previous_total = dividends.loc[
        (dividends.index > previous_start)
        & (dividends.index <= current_start)
    ].sum()

    if previous_total <= 0:
        return np.nan

    return safe_float(current_total / previous_total - 1.0)


def calculate_monthly_distribution_table(
    history: pd.DataFrame,
    shares: float,
) -> pd.DataFrame:
    """
    Return monthly cash distributions for an ETF position.
    """

    if history.empty or "Dividends" not in history.columns:
        return pd.DataFrame(
            columns=[
                "Month",
                "Distribution Per Share",
                "Position Income",
            ]
        )

    dividends = pd.to_numeric(
        history["Dividends"],
        errors="coerce",
    ).fillna(0.0)

    dividends = dividends[dividends > 0]

    if dividends.empty:
        return pd.DataFrame(
            columns=[
                "Month",
                "Distribution Per Share",
                "Position Income",
            ]
        )

    monthly = dividends.resample("ME").sum()

    result = pd.DataFrame(
        {
            "Month": monthly.index,
            "Distribution Per Share": monthly.values,
        }
    )

    result["Position Income"] = (
        result["Distribution Per Share"] * shares
    )

    return result


def analyze_etf(
    ticker_symbol: str,
    investment: float,
    history_period: str,
) -> tuple[dict[str, Any] | None, pd.DataFrame, str | None]:
    """
    Analyze one ETF.

    Returns:
        1. Summary metrics dictionary
        2. Monthly distribution table
        3. Error message, if any
    """

    try:
        history = download_ticker_history(
            ticker_symbol=ticker_symbol,
            period=history_period,
        )

        if history.empty:
            return (
                None,
                pd.DataFrame(),
                f"No market data was returned for {ticker_symbol}.",
            )

        price_column = choose_price_column(history)

        if price_column is None:
            return (
                None,
                pd.DataFrame(),
                f"No usable price column was returned for {ticker_symbol}.",
            )

        prices = pd.to_numeric(
            history[price_column],
            errors="coerce",
        ).dropna()

        if prices.empty:
            return (
                None,
                pd.DataFrame(),
                f"No valid prices were returned for {ticker_symbol}.",
            )

        current_price = safe_float(prices.iloc[-1])

        if not np.isfinite(current_price) or current_price <= 0:
            return (
                None,
                pd.DataFrame(),
                f"Invalid current price returned for {ticker_symbol}.",
            )

        shares = investment / current_price

        if "Dividends" in history.columns:
            dividends = pd.to_numeric(
                history["Dividends"],
                errors="coerce",
            ).fillna(0.0)

            dividends = dividends[dividends > 0]

        else:
            dividends = pd.Series(
                dtype=float,
                index=pd.DatetimeIndex([]),
            )

        latest_market_date = prices.index.max()
        trailing_start = latest_market_date - pd.Timedelta(days=365)

        trailing_dividends = dividends.loc[
            dividends.index > trailing_start
        ]

        trailing_distribution_per_share = safe_float(
            trailing_dividends.sum()
        )

        if not np.isfinite(trailing_distribution_per_share):
            trailing_distribution_per_share = 0.0

        trailing_yield = (
            trailing_distribution_per_share / current_price
            if current_price > 0
            else np.nan
        )

        estimated_annual_income = (
            trailing_distribution_per_share * shares
        )

        estimated_monthly_income = (
            estimated_annual_income / 12.0
        )

        latest_distribution = (
            safe_float(dividends.iloc[-1])
            if not dividends.empty
            else np.nan
        )

        latest_distribution_date = (
            dividends.index[-1]
            if not dividends.empty
            else pd.NaT
        )

        estimated_latest_payment = (
            latest_distribution * shares
            if np.isfinite(latest_distribution)
            else np.nan
        )

        one_year_start = latest_market_date - pd.Timedelta(days=365)
        three_year_start = latest_market_date - pd.Timedelta(
            days=365 * 3
        )

        one_year_price_return = calculate_price_return(
            prices,
            one_year_start,
        )

        three_year_price_return = calculate_price_return(
            prices,
            three_year_start,
        )

        max_drawdown = calculate_max_drawdown(prices)
        annualized_volatility = calculate_annualized_volatility(
            prices
        )

        distribution_growth = calculate_distribution_growth(
            dividends
        )

        metadata = download_ticker_metadata(ticker_symbol)

        fund_name = (
            metadata.get("longName")
            or metadata.get("shortName")
            or ETF_NAMES.get(ticker_symbol)
            or ticker_symbol
        )

        net_assets = safe_float(
            metadata.get("totalAssets", np.nan)
        )

        expense_ratio = safe_float(
            metadata.get("annualReportExpenseRatio", np.nan)
        )

        if (
            np.isfinite(expense_ratio)
            and expense_ratio > 1
        ):
            expense_ratio = expense_ratio / 100.0

        monthly_distributions = (
            calculate_monthly_distribution_table(
                history=history,
                shares=shares,
            )
        )

        summary = {
            "Ticker": ticker_symbol,
            "ETF Name": fund_name,
            "Current Price": current_price,
            "Investment": investment,
            "Shares": shares,
            "TTM Distribution Per Share":
                trailing_distribution_per_share,
            "TTM Distribution Yield": trailing_yield,
            "Estimated Monthly Income":
                estimated_monthly_income,
            "Estimated Annual Income":
                estimated_annual_income,
            "Latest Distribution Per Share":
                latest_distribution,
            "Estimated Latest Payment":
                estimated_latest_payment,
            "Latest Distribution Date":
                latest_distribution_date,
            "Distribution Growth":
                distribution_growth,
            "1-Year Price Return":
                one_year_price_return,
            "3-Year Price Return":
                three_year_price_return,
            "Maximum Drawdown":
                max_drawdown,
            "Annualized Volatility":
                annualized_volatility,
            "Net Assets": net_assets,
            "Expense Ratio": expense_ratio,
            "History": history,
        }

        return summary, monthly_distributions, None

    except Exception as exc:
        return (
            None,
            pd.DataFrame(),
            f"{ticker_symbol}: {type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("ETF Settings")

    ticker_text = st.text_area(
        "ETF tickers",
        value=", ".join(DEFAULT_TICKERS),
        height=130,
        help=(
            "Enter ticker symbols separated by commas, spaces, "
            "semicolons, or new lines."
        ),
    )

    investment_per_etf = st.number_input(
        "Investment per ETF",
        min_value=1_000.0,
        max_value=10_000_000.0,
        value=DEFAULT_INVESTMENT_PER_ETF,
        step=10_000.0,
        format="%.2f",
    )

    history_period = st.selectbox(
        "Historical period",
        options=["1y", "2y", "5y", "10y", "max"],
        index=2,
        help=(
            "A longer history improves drawdown and volatility "
            "analysis but may take longer to download."
        ),
    )

    minimum_yield = st.slider(
        "Minimum trailing yield",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.5,
        help="Filter the displayed results by trailing 12-month yield.",
    )

    sort_by = st.selectbox(
        "Sort results by",
        options=[
            "Estimated Monthly Income",
            "TTM Distribution Yield",
            "Estimated Annual Income",
            "1-Year Price Return",
            "Maximum Drawdown",
            "Annualized Volatility",
            "Ticker",
        ],
        index=0,
    )

    ascending_sort = st.checkbox(
        "Sort ascending",
        value=False,
    )

    analyze_button = st.button(
        "Analyze ETFs",
        type="primary",
        use_container_width=True,
    )

    if st.button(
        "Clear downloaded data",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.success("Cached data cleared.")


# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------

st.title("High-Yield ETF Income Analyzer")

st.write(
    "Compare covered-call and high-distribution ETFs using an assumed "
    f"investment of **{format_large_currency(investment_per_etf)} "
    "in each fund**."
)

st.caption(
    "Monthly income is estimated from the ETF's trailing 12-month "
    "cash distributions divided by 12. Actual distributions can vary "
    "substantially from month to month."
)


# ---------------------------------------------------------------------
# VALIDATE INPUT
# ---------------------------------------------------------------------

tickers = parse_tickers(ticker_text)

if not tickers:
    st.error("Enter at least one valid ETF ticker.")
    st.stop()

if len(tickers) > 30:
    st.warning(
        "For performance, the app analyzes only the first 30 tickers."
    )
    tickers = tickers[:30]


# Run automatically on the first page load as well as when button is used.
if "analysis_requested" not in st.session_state:
    st.session_state.analysis_requested = True

if analyze_button:
    st.session_state.analysis_requested = True


# ---------------------------------------------------------------------
# RUN ANALYSIS
# ---------------------------------------------------------------------

summaries: list[dict[str, Any]] = []
monthly_data_frames: list[pd.DataFrame] = []
errors: list[str] = []

progress_bar = st.progress(0)
status_placeholder = st.empty()

for index, ticker_symbol in enumerate(tickers):
    status_placeholder.write(
        f"Analyzing **{ticker_symbol}**..."
    )

    summary, monthly_table, error = analyze_etf(
        ticker_symbol=ticker_symbol,
        investment=investment_per_etf,
        history_period=history_period,
    )

    if error:
        errors.append(error)

    if summary is not None:
        summaries.append(summary)

        if not monthly_table.empty:
            monthly_table = monthly_table.copy()
            monthly_table["Ticker"] = ticker_symbol
            monthly_data_frames.append(monthly_table)

    progress_bar.progress((index + 1) / len(tickers))

progress_bar.empty()
status_placeholder.empty()


# ---------------------------------------------------------------------
# HANDLE FAILED DOWNLOADS
# ---------------------------------------------------------------------

if errors:
    with st.expander(
        f"Data warnings ({len(errors)})",
        expanded=not summaries,
    ):
        for error in errors:
            st.warning(error)

if not summaries:
    st.error(
        "No ETF data was returned. Check the symbols and try again."
    )
    st.stop()


# ---------------------------------------------------------------------
# CREATE SUMMARY DATAFRAME
# ---------------------------------------------------------------------

results_df = pd.DataFrame(
    [
        {
            key: value
            for key, value in summary.items()
            if key != "History"
        }
        for summary in summaries
    ]
)

results_df = results_df[
    results_df["TTM Distribution Yield"].fillna(0)
    >= minimum_yield / 100.0
].copy()

if results_df.empty:
    st.warning(
        "No ETFs meet the selected minimum-yield requirement."
    )
    st.stop()

if sort_by in results_df.columns:
    results_df.sort_values(
        by=sort_by,
        ascending=ascending_sort,
        inplace=True,
        na_position="last",
    )


# ---------------------------------------------------------------------
# PORTFOLIO SUMMARY
# ---------------------------------------------------------------------

total_investment = safe_float(
    results_df["Investment"].sum()
)

total_monthly_income = safe_float(
    results_df["Estimated Monthly Income"].sum()
)

total_annual_income = safe_float(
    results_df["Estimated Annual Income"].sum()
)

portfolio_yield = (
    total_annual_income / total_investment
    if total_investment > 0
    else np.nan
)

average_monthly_income_per_etf = safe_float(
    results_df["Estimated Monthly Income"].mean()
)

st.subheader("Portfolio Income Summary")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric(
    "Total Investment",
    format_large_currency(total_investment),
)

metric_col2.metric(
    "Estimated Monthly Income",
    format_currency(total_monthly_income),
)

metric_col3.metric(
    "Estimated Annual Income",
    format_currency(total_annual_income),
)

metric_col4.metric(
    "Portfolio Distribution Yield",
    format_percentage(portfolio_yield),
)

st.caption(
    f"Average estimated monthly income per ETF: "
    f"{format_currency(average_monthly_income_per_etf)}"
)


# ---------------------------------------------------------------------
# RESULTS TABLE
# ---------------------------------------------------------------------

st.subheader("ETF Income Comparison")

display_columns = [
    "Ticker",
    "ETF Name",
    "Current Price",
    "Investment",
    "Shares",
    "TTM Distribution Per Share",
    "TTM Distribution Yield",
    "Estimated Monthly Income",
    "Estimated Annual Income",
    "Latest Distribution Per Share",
    "Estimated Latest Payment",
    "Latest Distribution Date",
    "Distribution Growth",
    "1-Year Price Return",
    "3-Year Price Return",
    "Maximum Drawdown",
    "Annualized Volatility",
    "Expense Ratio",
]

available_display_columns = [
    column
    for column in display_columns
    if column in results_df.columns
]

table_df = results_df[available_display_columns].copy()

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Current Price": st.column_config.NumberColumn(
            "Current Price",
            format="$%.2f",
        ),
        "Investment": st.column_config.NumberColumn(
            "Investment",
            format="$%.0f",
        ),
        "Shares": st.column_config.NumberColumn(
            "Shares Purchased",
            format="%.2f",
        ),
        "TTM Distribution Per Share":
            st.column_config.NumberColumn(
                "TTM Distribution/Share",
                format="$%.4f",
            ),
        "TTM Distribution Yield":
            st.column_config.NumberColumn(
                "TTM Yield",
                format="%.2f%%",
            ),
        "Estimated Monthly Income":
            st.column_config.NumberColumn(
                "Estimated Monthly Income",
                format="$%.2f",
            ),
        "Estimated Annual Income":
            st.column_config.NumberColumn(
                "Estimated Annual Income",
                format="$%.2f",
            ),
        "Latest Distribution Per Share":
            st.column_config.NumberColumn(
                "Latest Distribution/Share",
                format="$%.4f",
            ),
        "Estimated Latest Payment":
            st.column_config.NumberColumn(
                "Estimated Latest Payment",
                format="$%.2f",
            ),
        "Latest Distribution Date":
            st.column_config.DateColumn(
                "Latest Distribution Date",
                format="MMM D, YYYY",
            ),
        "Distribution Growth":
            st.column_config.NumberColumn(
                "TTM Distribution Growth",
                format="%.2f%%",
            ),
        "1-Year Price Return":
            st.column_config.NumberColumn(
                "1-Year Price Return",
                format="%.2f%%",
            ),
        "3-Year Price Return":
            st.column_config.NumberColumn(
                "3-Year Price Return",
                format="%.2f%%",
            ),
        "Maximum Drawdown":
            st.column_config.NumberColumn(
                "Maximum Drawdown",
                format="%.2f%%",
            ),
        "Annualized Volatility":
            st.column_config.NumberColumn(
                "Annualized Volatility",
                format="%.2f%%",
            ),
        "Expense Ratio":
            st.column_config.NumberColumn(
                "Expense Ratio",
                format="%.2f%%",
            ),
    },
)


# ---------------------------------------------------------------------
# DOWNLOADABLE CSV
# ---------------------------------------------------------------------

csv_df = results_df.copy()

if "Latest Distribution Date" in csv_df.columns:
    csv_df["Latest Distribution Date"] = pd.to_datetime(
        csv_df["Latest Distribution Date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

csv_data = csv_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download ETF analysis as CSV",
    data=csv_data,
    file_name=(
        f"high_yield_etf_analysis_"
        f"{datetime.now():%Y-%m-%d}.csv"
    ),
    mime="text/csv",
)


# ---------------------------------------------------------------------
# INCOME CHARTS
# ---------------------------------------------------------------------

st.subheader("Estimated Income")

monthly_income_chart = px.bar(
    results_df,
    x="Ticker",
    y="Estimated Monthly Income",
    text_auto=".2s",
    title=(
        f"Average Monthly Income Based on "
        f"{format_large_currency(investment_per_etf)} per ETF"
    ),
    labels={
        "Ticker": "ETF",
        "Estimated Monthly Income": "Estimated Monthly Income",
    },
)

monthly_income_chart.update_layout(
    yaxis_tickprefix="$",
    hovermode="x unified",
)

st.plotly_chart(
    monthly_income_chart,
    use_container_width=True,
)

yield_chart = px.bar(
    results_df,
    x="Ticker",
    y="TTM Distribution Yield",
    text_auto=".2%",
    title="Trailing 12-Month Distribution Yield",
    labels={
        "Ticker": "ETF",
        "TTM Distribution Yield": "Trailing Yield",
    },
)

yield_chart.update_layout(
    yaxis_tickformat=".1%",
    hovermode="x unified",
)

st.plotly_chart(
    yield_chart,
    use_container_width=True,
)


# ---------------------------------------------------------------------
# RISK AND RETURN CHART
# ---------------------------------------------------------------------

st.subheader("Risk and Price Performance")

risk_chart_df = results_df.dropna(
    subset=[
        "Annualized Volatility",
        "1-Year Price Return",
    ]
).copy()

if not risk_chart_df.empty:
    risk_chart = px.scatter(
        risk_chart_df,
        x="Annualized Volatility",
        y="1-Year Price Return",
        size="Estimated Annual Income",
        hover_name="Ticker",
        hover_data={
            "ETF Name": True,
            "TTM Distribution Yield": ":.2%",
            "Estimated Monthly Income": ":$,.2f",
            "Annualized Volatility": ":.2%",
            "1-Year Price Return": ":.2%",
        },
        title="One-Year Price Return vs. Annualized Volatility",
        labels={
            "Annualized Volatility": "Annualized Volatility",
            "1-Year Price Return": "One-Year Price Return",
        },
    )

    risk_chart.update_layout(
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
    )

    st.plotly_chart(
        risk_chart,
        use_container_width=True,
    )

else:
    st.info(
        "Not enough historical data is available for the "
        "risk-and-return chart."
    )


# ---------------------------------------------------------------------
# HISTORICAL MONTHLY DISTRIBUTIONS
# ---------------------------------------------------------------------

st.subheader("Historical Monthly Cash Distributions")

if monthly_data_frames:
    all_monthly_distributions = pd.concat(
        monthly_data_frames,
        ignore_index=True,
    )

    selected_history_tickers = st.multiselect(
        "ETFs to show in the distribution chart",
        options=results_df["Ticker"].tolist(),
        default=results_df["Ticker"].tolist()[:5],
    )

    months_to_show = st.slider(
        "Months of distribution history",
        min_value=6,
        max_value=60,
        value=24,
        step=1,
    )

    cutoff_date = (
        pd.Timestamp.today().normalize()
        - pd.DateOffset(months=months_to_show)
    )

    chart_monthly_df = all_monthly_distributions[
        (
            all_monthly_distributions["Ticker"].isin(
                selected_history_tickers
            )
        )
        & (
            all_monthly_distributions["Month"] >= cutoff_date
        )
    ].copy()

    if not chart_monthly_df.empty:
        distribution_chart = px.line(
            chart_monthly_df,
            x="Month",
            y="Position Income",
            color="Ticker",
            markers=True,
            title=(
                "Actual Historical Monthly Cash Distributions "
                "for Each Modeled Position"
            ),
            labels={
                "Month": "Month",
                "Position Income": "Position Income",
                "Ticker": "ETF",
            },
        )

        distribution_chart.update_layout(
            yaxis_tickprefix="$",
            hovermode="x unified",
        )

        st.plotly_chart(
            distribution_chart,
            use_container_width=True,
        )

    else:
        st.info(
            "No monthly distribution data is available for the "
            "selected ETFs and date range."
        )

else:
    st.info(
        "Yahoo Finance did not return distribution history "
        "for the selected ETFs."
    )


# ---------------------------------------------------------------------
# NORMALIZED PRICE CHART
# ---------------------------------------------------------------------

st.subheader("Normalized Price Performance")

selected_price_tickers = st.multiselect(
    "ETFs to show in the price chart",
    options=results_df["Ticker"].tolist(),
    default=results_df["Ticker"].tolist()[:5],
    key="price_chart_tickers",
)

normalized_frames: list[pd.DataFrame] = []

summary_lookup = {
    summary["Ticker"]: summary
    for summary in summaries
}

for ticker_symbol in selected_price_tickers:
    summary = summary_lookup.get(ticker_symbol)

    if not summary:
        continue

    history = summary.get("History")

    if history is None or history.empty:
        continue

    price_column = choose_price_column(history)

    if price_column is None:
        continue

    prices = pd.to_numeric(
        history[price_column],
        errors="coerce",
    ).dropna()

    if prices.empty:
        continue

    normalized = prices / prices.iloc[0] * 100.0

    normalized_frame = pd.DataFrame(
        {
            "Date": normalized.index,
            "Normalized Price": normalized.values,
            "Ticker": ticker_symbol,
        }
    )

    normalized_frames.append(normalized_frame)

if normalized_frames:
    normalized_price_df = pd.concat(
        normalized_frames,
        ignore_index=True,
    )

    normalized_chart = px.line(
        normalized_price_df,
        x="Date",
        y="Normalized Price",
        color="Ticker",
        title="Price Growth of $100 Initial Value",
        labels={
            "Date": "Date",
            "Normalized Price": "Normalized Price",
            "Ticker": "ETF",
        },
    )

    normalized_chart.add_hline(
        y=100,
        line_dash="dash",
    )

    normalized_chart.update_layout(
        hovermode="x unified",
    )

    st.plotly_chart(
        normalized_chart,
        use_container_width=True,
    )

else:
    st.info(
        "Select at least one ETF with sufficient price history."
    )


# ---------------------------------------------------------------------
# ETF DETAIL SECTION
# ---------------------------------------------------------------------

st.subheader("ETF Details")

detail_ticker = st.selectbox(
    "Select an ETF",
    options=results_df["Ticker"].tolist(),
)

detail_row = results_df.loc[
    results_df["Ticker"] == detail_ticker
].iloc[0]

detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)

detail_col1.metric(
    "Investment",
    format_large_currency(detail_row["Investment"]),
)

detail_col2.metric(
    "Shares Purchased",
    f"{detail_row['Shares']:,.2f}",
)

detail_col3.metric(
    "Estimated Monthly Income",
    format_currency(
        detail_row["Estimated Monthly Income"]
    ),
)

detail_col4.metric(
    "Estimated Annual Income",
    format_currency(
        detail_row["Estimated Annual Income"]
    ),
)

st.write(
    f"**Fund:** {detail_row['ETF Name']}"
)

st.write(
    f"**Trailing distribution yield:** "
    f"{format_percentage(detail_row['TTM Distribution Yield'])}"
)

st.write(
    f"**Latest estimated cash payment on the modeled position:** "
    f"{format_currency(detail_row['Estimated Latest Payment'])}"
)

if pd.notna(detail_row["Latest Distribution Date"]):
    latest_date_text = pd.Timestamp(
        detail_row["Latest Distribution Date"]
    ).strftime("%B %d, %Y")

    st.write(
        f"**Latest distribution date in the downloaded data:** "
        f"{latest_date_text}"
    )


# ---------------------------------------------------------------------
# METHODOLOGY
# ---------------------------------------------------------------------

with st.expander("How the income estimate is calculated"):
    st.markdown(
        """
### Position size

The app assumes the selected investment amount is placed into each ETF.

**Shares purchased**

Investment divided by the latest available closing price.

### Trailing distribution yield

The app totals all cash distributions recorded during the latest
365-day period and divides that total by the latest closing price.

### Estimated annual income

Shares purchased multiplied by the trailing 12-month distributions
per share.

### Estimated monthly income

Estimated annual income divided by 12.

This is an average—not a prediction that every future monthly payment
will be identical.

Some ETFs distribute monthly, some weekly, and some on other
schedules. Covered-call ETF distributions can change substantially
with market volatility, option premiums, realized gains, fund policy,
and return-of-capital treatment.
"""
    )


# ---------------------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------------------

st.divider()

st.caption(
    "Educational use only. This application does not provide investment, "
    "tax, accounting, or legal advice. Distribution yield is not the same "
    "as total return. Covered-call funds may sacrifice price appreciation, "
    "experience capital losses, reduce distributions, or classify part of "
    "a distribution as return of capital. Data from Yahoo Finance may be "
    "delayed, incomplete, or revised."
)
