import math
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ---------------------------------------------------------
# App configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="High-Yield Covered-Call ETF Scanner",
    page_icon="💵",
    layout="wide",
)


DEFAULT_TICKERS = [
    "QYLD",
    "XYLD",
    "RYLD",
    "QYLG",
    "XYLG",
    "JEPI",
    "JEPQ",
    "SPYI",
    "QQQI",
    "DIVO",
    "IDVO",
    "GPIX",
    "GPIQ",
    "QDTE",
    "XDTE",
    "RDTE",
    "NUSI",
]

DEFAULT_BASKET = {
    "QQQI": 30.0,
    "JEPQ": 30.0,
    "SPYI": 20.0,
    "JEPI": 20.0,
}


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def normalize_tickers(raw_text: str) -> list[str]:
    """Return unique uppercase ticker symbols from free-form text."""
    cleaned = (
        raw_text.replace("\n", ",")
        .replace(";", ",")
        .replace(" ", ",")
    )

    tickers = [
        item.strip().upper()
        for item in cleaned.split(",")
        if item.strip()
    ]

    return list(dict.fromkeys(tickers))


@st.cache_data(ttl=3600, show_spinner=False)
def load_history(ticker: str, period: str) -> pd.DataFrame:
    """
    Load ETF price and distribution history.

    auto_adjust=False is required because:
    - Close is used for current market price.
    - Adj Close is used for total-return calculations.
    - Dividends are analyzed separately.
    """
    history = yf.Ticker(ticker).history(
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=True,
        repair=True,
    )

    if history.empty:
        return pd.DataFrame()

    history = history.copy()

    index = pd.to_datetime(history.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    history.index = index

    if "Close" not in history.columns:
        history["Close"] = np.nan

    if "Adj Close" not in history.columns:
        history["Adj Close"] = history["Close"]

    if "Dividends" not in history.columns:
        history["Dividends"] = 0.0

    history["Close"] = pd.to_numeric(
        history["Close"],
        errors="coerce",
    )

    history["Adj Close"] = pd.to_numeric(
        history["Adj Close"],
        errors="coerce",
    )

    history["Dividends"] = (
        pd.to_numeric(history["Dividends"], errors="coerce")
        .fillna(0.0)
    )

    return history


def annualized_return(prices: pd.Series) -> float:
    """Calculate CAGR for an adjusted-price series."""
    prices = prices.dropna()

    if len(prices) < 2:
        return np.nan

    first_price = float(prices.iloc[0])
    last_price = float(prices.iloc[-1])

    if first_price <= 0 or last_price <= 0:
        return np.nan

    elapsed_days = (prices.index[-1] - prices.index[0]).days
    years = elapsed_days / 365.25

    if years <= 0:
        return np.nan

    return (last_price / first_price) ** (1 / years) - 1


def annualized_volatility(prices: pd.Series) -> float:
    """Calculate annualized volatility from daily adjusted returns."""
    daily_returns = prices.dropna().pct_change().dropna()

    if len(daily_returns) < 20:
        return np.nan

    return float(daily_returns.std() * math.sqrt(252))


def maximum_drawdown(prices: pd.Series) -> float:
    """Calculate maximum peak-to-trough decline."""
    prices = prices.dropna()

    if prices.empty:
        return np.nan

    running_peak = prices.cummax()
    drawdowns = prices / running_peak - 1

    return float(drawdowns.min())


def distribution_growth(dividends: pd.Series) -> float:
    """
    Compare the latest 12 months of distributions with the preceding
    12-month period.
    """
    dividends = dividends.dropna()

    if dividends.empty:
        return np.nan

    latest_date = dividends.index.max()
    latest_period_start = latest_date - pd.DateOffset(years=1)
    previous_period_start = latest_date - pd.DateOffset(years=2)

    latest_total = dividends.loc[
        dividends.index > latest_period_start
    ].sum()

    previous_total = dividends.loc[
        (dividends.index > previous_period_start)
        & (dividends.index <= latest_period_start)
    ].sum()

    if previous_total <= 0:
        return np.nan

    return float(latest_total / previous_total - 1)


def analyze_ticker(
    ticker: str,
    period: str,
) -> tuple[dict | None, str | None]:
    """Calculate price, income, return, risk, and distribution metrics."""
    try:
        history = load_history(ticker, period)

        if history.empty:
            return None, "No history returned by Yahoo Finance"

        close_prices = history["Close"].dropna()

        if close_prices.empty:
            return None, "No closing-price data returned"

        latest_date = close_prices.index[-1]
        current_price = float(close_prices.iloc[-1])

        if current_price <= 0:
            return None, "Invalid current price"

        trailing_start = latest_date - pd.DateOffset(years=1)

        trailing_distributions = float(
            history.loc[
                history.index > trailing_start,
                "Dividends",
            ].sum()
        )

        trailing_yield = trailing_distributions / current_price

        adjusted_prices = history["Adj Close"].dropna()

        if adjusted_prices.empty:
            adjusted_prices = close_prices

        one_year_prices = adjusted_prices.loc[
            adjusted_prices.index > trailing_start
        ]

        if len(one_year_prices) >= 2:
            one_year_total_return = float(
                one_year_prices.iloc[-1]
                / one_year_prices.iloc[0]
                - 1
            )
        else:
            one_year_total_return = np.nan

        recent_history = history.loc[
            history.index > trailing_start
        ].copy()

        recent_history["Payment Month"] = (
            recent_history.index.to_period("M")
        )

        monthly_distributions = (
            recent_history.groupby("Payment Month")["Dividends"]
            .sum()
        )

        months_paid = int(
            (monthly_distributions > 0).sum()
        )

        average_payment = (
            trailing_distributions / months_paid
            if months_paid > 0
            else 0.0
        )

        metrics = {
            "Ticker": ticker,
            "Price": current_price,
            "TTM Distribution": trailing_distributions,
            "TTM Yield": trailing_yield,
            "1Y Total Return": one_year_total_return,
            "Annualized Return": annualized_return(
                adjusted_prices
            ),
            "Volatility": annualized_volatility(
                adjusted_prices
            ),
            "Max Drawdown": maximum_drawdown(
                adjusted_prices
            ),
            "Distribution Growth": distribution_growth(
                history["Dividends"]
            ),
            "Months Paid": months_paid,
            "Average Payment": average_payment,
            "History Start": history.index.min().date(),
            "As Of": latest_date.date(),
        }

        return metrics, None

    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def percentile_score(
    frame: pd.DataFrame,
    yield_weight: float,
    return_weight: float,
    drawdown_weight: float,
    volatility_weight: float,
) -> pd.Series:
    """
    Rank ETFs using percentile ranks.

    Higher is better for:
    - TTM yield
    - 1-year total return
    - Max drawdown, because -10% is better than -40%

    Lower is better for:
    - Volatility
    """
    total_weight = (
        yield_weight
        + return_weight
        + drawdown_weight
        + volatility_weight
    )

    if total_weight <= 0:
        return pd.Series(
            0.0,
            index=frame.index,
            dtype=float,
        )

    yield_rank = frame["TTM Yield"].rank(
        pct=True,
        na_option="bottom",
    )

    return_rank = frame["1Y Total Return"].rank(
        pct=True,
        na_option="bottom",
    )

    drawdown_rank = frame["Max Drawdown"].rank(
        pct=True,
        na_option="bottom",
    )

    volatility_rank = (
        -frame["Volatility"]
    ).rank(
        pct=True,
        na_option="bottom",
    )

    score = (
        yield_rank * yield_weight
        + return_rank * return_weight
        + drawdown_rank * drawdown_weight
        + volatility_rank * volatility_weight
    ) / total_weight

    return score * 100


def build_portfolio_table(
    results: pd.DataFrame,
    weights: dict[str, float],
    investment_amount: float,
) -> pd.DataFrame:
    """Build allocation and estimated-income table."""
    entered_total = sum(
        max(weight, 0.0)
        for weight in weights.values()
    )

    if entered_total <= 0:
        return pd.DataFrame()

    indexed_results = results.set_index("Ticker")
    rows = []

    for ticker, raw_weight in weights.items():
        if ticker not in indexed_results.index:
            continue

        if raw_weight <= 0:
            continue

        normalized_weight = raw_weight / entered_total
        allocation = investment_amount * normalized_weight
        ticker_data = indexed_results.loc[ticker]

        estimated_annual_income = (
            allocation * ticker_data["TTM Yield"]
        )

        rows.append(
            {
                "Ticker": ticker,
                "Weight": normalized_weight,
                "Allocation": allocation,
                "TTM Yield": ticker_data["TTM Yield"],
                "Estimated Annual Income": (
                    estimated_annual_income
                ),
                "Estimated Monthly Income": (
                    estimated_annual_income / 12
                ),
                "1Y Total Return": (
                    ticker_data["1Y Total Return"]
                ),
                "Max Drawdown": (
                    ticker_data["Max Drawdown"]
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Page heading
# ---------------------------------------------------------

st.title("💵 High-Yield Covered-Call ETF Scanner")

st.caption(
    "Compare covered-call and option-income ETFs using "
    "distribution yield, total return, drawdown, volatility, "
    "and distribution history."
)


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------

with st.sidebar:
    st.header("Scanner Settings")

    ticker_text = st.text_area(
        "Candidate tickers",
        value=", ".join(DEFAULT_TICKERS),
        height=180,
        help=(
            "Enter symbols separated by commas, spaces, "
            "semicolons, or new lines."
        ),
    )

    history_period = st.selectbox(
        "Price-history period",
        options=[
            "1y",
            "2y",
            "5y",
            "10y",
            "max",
        ],
        index=1,
        help=(
            "Only Yahoo Finance-supported periods are used. "
            "Newer ETFs may not have long histories."
        ),
    )

    minimum_yield = (
        st.slider(
            "Minimum trailing yield",
            min_value=0.0,
            max_value=40.0,
            value=7.0,
            step=0.5,
            format="%.1f%%",
        )
        / 100
    )

    maximum_allowed_drawdown = (
        st.slider(
            "Worst acceptable drawdown",
            min_value=-80.0,
            max_value=0.0,
            value=-45.0,
            step=1.0,
            format="%.0f%%",
            help=(
                "An ETF with a historical drawdown below this "
                "threshold is excluded."
            ),
        )
        / 100
    )

    st.subheader("Ranking Weights")

    yield_weight = st.slider(
        "Yield weight",
        min_value=0,
        max_value=100,
        value=35,
    )

    return_weight = st.slider(
        "Total-return weight",
        min_value=0,
        max_value=100,
        value=30,
    )

    drawdown_weight = st.slider(
        "Drawdown-protection weight",
        min_value=0,
        max_value=100,
        value=20,
    )

    volatility_weight = st.slider(
        "Low-volatility weight",
        min_value=0,
        max_value=100,
        value=15,
    )

    run_scan = st.button(
        "Run ETF Scan",
        type="primary",
        use_container_width=True,
    )

    clear_cache = st.button(
        "Clear Cached Market Data",
        use_container_width=True,
    )

    if clear_cache:
        st.cache_data.clear()
        st.session_state.pop("scan_results", None)
        st.session_state.pop("scan_errors", None)
        st.success("Cache cleared.")


# ---------------------------------------------------------
# Run scan
# ---------------------------------------------------------

if run_scan or "scan_results" not in st.session_state:
    ticker_list = normalize_tickers(ticker_text)

    rows = []
    errors = {}

    if not ticker_list:
        st.error("Enter at least one ETF ticker.")
        st.stop()

    progress_bar = st.progress(
        0,
        text="Starting ETF scan...",
    )

    for position, ticker in enumerate(
        ticker_list,
        start=1,
    ):
        row, error = analyze_ticker(
            ticker=ticker,
            period=history_period,
        )

        if row is not None:
            rows.append(row)

        if error is not None:
            errors[ticker] = error

        progress_bar.progress(
            position / len(ticker_list),
            text=f"Analyzing {ticker}...",
        )

    progress_bar.empty()

    st.session_state["scan_results"] = pd.DataFrame(
        rows
    )

    st.session_state["scan_errors"] = errors


results = st.session_state.get(
    "scan_results",
    pd.DataFrame(),
).copy()

errors = st.session_state.get(
    "scan_errors",
    {},
)


# ---------------------------------------------------------
# Empty-data handling
# ---------------------------------------------------------

if results.empty:
    st.error("No ETF data was returned.")

    st.info(
        "Possible causes include temporary Yahoo Finance "
        "rate limiting, an outbound-network restriction, "
        "or invalid symbols."
    )

    if errors:
        st.subheader("Download Errors")
        st.json(errors)

    st.stop()


# ---------------------------------------------------------
# Ranking and filtering
# ---------------------------------------------------------

results["Score"] = percentile_score(
    frame=results,
    yield_weight=yield_weight,
    return_weight=return_weight,
    drawdown_weight=drawdown_weight,
    volatility_weight=volatility_weight,
)

filtered_results = results.loc[
    (results["TTM Yield"] >= minimum_yield)
    & (
        results["Max Drawdown"]
        >= maximum_allowed_drawdown
    )
].copy()

filtered_results = filtered_results.sort_values(
    by=[
        "Score",
        "TTM Yield",
    ],
    ascending=[
        False,
        False,
    ],
)


# ---------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "ETFs Analyzed",
    f"{len(results)}",
)

metric_2.metric(
    "Passing Filters",
    f"{len(filtered_results)}",
)

metric_3.metric(
    "Highest TTM Yield",
    f"{results['TTM Yield'].max():.2%}",
)

metric_4.metric(
    "Best Scanner Score",
    f"{results['Score'].max():.1f}",
)


# ---------------------------------------------------------
# Scanner table
# ---------------------------------------------------------

st.subheader("Ranked ETF Candidates")

display_columns = [
    "Ticker",
    "Score",
    "Price",
    "TTM Yield",
    "1Y Total Return",
    "Annualized Return",
    "Max Drawdown",
    "Volatility",
    "Distribution Growth",
    "Months Paid",
    "TTM Distribution",
    "Average Payment",
    "History Start",
    "As Of",
]

if filtered_results.empty:
    st.warning(
        "No ETFs passed the current yield and drawdown filters. "
        "Try lowering the minimum yield or allowing a larger drawdown."
    )

    table_data = results.sort_values(
        by="Score",
        ascending=False,
    )
else:
    table_data = filtered_results

st.dataframe(
    table_data[display_columns],
    hide_index=True,
    use_container_width=True,
    column_config={
        "Score": st.column_config.ProgressColumn(
            "Score",
            min_value=0,
            max_value=100,
            format="%.1f",
        ),
        "Price": st.column_config.NumberColumn(
            "Price",
            format="$%.2f",
        ),
        "TTM Yield": st.column_config.NumberColumn(
            "TTM Yield",
            format="%.2f%%",
        ),
        "1Y Total Return": st.column_config.NumberColumn(
            "1Y Total Return",
            format="%.2f%%",
        ),
        "Annualized Return": (
            st.column_config.NumberColumn(
                "Annualized Return",
                format="%.2f%%",
            )
        ),
        "Max Drawdown": st.column_config.NumberColumn(
            "Max Drawdown",
            format="%.2f%%",
        ),
        "Volatility": st.column_config.NumberColumn(
            "Volatility",
            format="%.2f%%",
        ),
        "Distribution Growth": (
            st.column_config.NumberColumn(
                "Distribution Growth",
                format="%.2f%%",
            )
        ),
        "TTM Distribution": (
            st.column_config.NumberColumn(
                "TTM Distribution",
                format="$%.4f",
            )
        ),
        "Average Payment": (
            st.column_config.NumberColumn(
                "Average Payment",
                format="$%.4f",
            )
        ),
    },
)

csv_data = table_data[
    display_columns
].to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Results as CSV",
    data=csv_data,
    file_name=(
        f"covered_call_etf_scan_"
        f"{date.today().isoformat()}.csv"
    ),
    mime="text/csv",
)


if errors:
    with st.expander(
        f"Ticker Download Errors ({len(errors)})"
    ):
        st.json(errors)


# ---------------------------------------------------------
# Basket builder
# ---------------------------------------------------------

st.divider()
st.header("Income Basket Builder")

available_tickers = sorted(
    results["Ticker"].dropna().unique().tolist()
)

default_selected = [
    ticker
    for ticker in DEFAULT_BASKET
    if ticker in available_tickers
]

selected_tickers = st.multiselect(
    "Choose ETFs for the basket",
    options=available_tickers,
    default=default_selected,
)

portfolio_amount = st.number_input(
    "Portfolio amount",
    min_value=1000.0,
    value=100000.0,
    step=5000.0,
    format="%.2f",
)

portfolio_weights = {}

if selected_tickers:
    st.write("Set target weights:")

    weight_columns = st.columns(
        min(len(selected_tickers), 4)
    )

    for index, ticker in enumerate(
        selected_tickers
    ):
        default_weight = DEFAULT_BASKET.get(
            ticker,
            100.0 / len(selected_tickers),
        )

        column = weight_columns[
            index % len(weight_columns)
        ]

        with column:
            portfolio_weights[ticker] = (
                st.number_input(
                    label=ticker,
                    min_value=0.0,
                    max_value=100.0,
                    value=float(default_weight),
                    step=5.0,
                    format="%.1f",
                    key=f"weight_{ticker}",
                )
            )

portfolio_table = build_portfolio_table(
    results=results,
    weights=portfolio_weights,
    investment_amount=portfolio_amount,
)

if not portfolio_table.empty:
    entered_weight_total = sum(
        portfolio_weights.values()
    )

    if not np.isclose(
        entered_weight_total,
        100.0,
    ):
        st.info(
            f"Entered weights total "
            f"{entered_weight_total:.1f}%. "
            "The app normalized them to 100%."
        )

    total_annual_income = float(
        portfolio_table[
            "Estimated Annual Income"
        ].sum()
    )

    total_monthly_income = float(
        portfolio_table[
            "Estimated Monthly Income"
        ].sum()
    )

    portfolio_yield = (
        total_annual_income / portfolio_amount
        if portfolio_amount > 0
        else np.nan
    )

    valid_returns = portfolio_table[
        "1Y Total Return"
    ].fillna(0.0)

    weighted_one_year_return = float(
        np.average(
            valid_returns,
            weights=portfolio_table["Weight"],
        )
    )

    basket_metric_1, basket_metric_2, basket_metric_3, basket_metric_4 = (
        st.columns(4)
    )

    basket_metric_1.metric(
        "Portfolio TTM Yield",
        f"{portfolio_yield:.2%}",
    )

    basket_metric_2.metric(
        "Estimated Annual Income",
        f"${total_annual_income:,.0f}",
    )

    basket_metric_3.metric(
        "Estimated Monthly Income",
        f"${total_monthly_income:,.0f}",
    )

    basket_metric_4.metric(
        "Weighted 1Y Total Return",
        f"{weighted_one_year_return:.2%}",
    )

    st.dataframe(
        portfolio_table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Weight": st.column_config.NumberColumn(
                "Weight",
                format="%.1f%%",
            ),
            "Allocation": (
                st.column_config.NumberColumn(
                    "Allocation",
                    format="$%.2f",
                )
            ),
            "TTM Yield": (
                st.column_config.NumberColumn(
                    "TTM Yield",
                    format="%.2f%%",
                )
            ),
            "Estimated Annual Income": (
                st.column_config.NumberColumn(
                    "Estimated Annual Income",
                    format="$%.2f",
                )
            ),
            "Estimated Monthly Income": (
                st.column_config.NumberColumn(
                    "Estimated Monthly Income",
                    format="$%.2f",
                )
            ),
            "1Y Total Return": (
                st.column_config.NumberColumn(
                    "1Y Total Return",
                    format="%.2f%%",
                )
            ),
            "Max Drawdown": (
                st.column_config.NumberColumn(
                    "Max Drawdown",
                    format="%.2f%%",
                )
            ),
        },
    )

    allocation_chart = (
        portfolio_table.set_index("Ticker")[
            ["Allocation"]
        ]
    )

    st.subheader("Portfolio Allocation")
    st.bar_chart(allocation_chart)


# ---------------------------------------------------------
# Notes
# ---------------------------------------------------------

st.divider()

st.warning(
    "Trailing distribution yield is not guaranteed income. "
    "Covered-call ETF distributions may contain dividends, "
    "option premium, capital gains, and return of capital. "
    "Review each fund's issuer documents, fees, NAV history, "
    "distribution notices, and tax treatment."
)
