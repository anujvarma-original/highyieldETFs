
import math
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Covered-Call ETF Scanner",
    page_icon="💵",
    layout="wide",
)

DEFAULT_TICKERS = [
    "QYLD", "XYLD", "RYLD",
    "JEPI", "JEPQ",
    "SPYI", "QQQI",
    "DIVO", "IDVO",
    "GPIX", "GPIQ",
    "QDTE", "XDTE", "RDTE",
    "QYLG", "XYLG",
    "NUSI",
]

DEFAULT_BASKET = {
    "QQQI": 30.0,
    "JEPQ": 30.0,
    "SPYI": 20.0,
    "JEPI": 20.0,
}


def normalize_tickers(raw_text: str) -> list[str]:
    """Convert comma/newline/space-separated input into unique uppercase tickers."""
    cleaned = raw_text.replace("\n", ",").replace(" ", ",")
    values = [x.strip().upper() for x in cleaned.split(",") if x.strip()]
    return list(dict.fromkeys(values))


@st.cache_data(ttl=3600, show_spinner=False)
def load_history(ticker: str, period: str) -> pd.DataFrame:
    """
    Download unadjusted prices plus distributions.

    auto_adjust=False is important because we separately use Adj Close for
    total-return calculations and Close for current-price/yield calculations.
    """
    data = yf.Ticker(ticker).history(
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=True,
        repair=True,
    )

    if data.empty:
        return data

    data = data.copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)

    for column in ["Close", "Adj Close", "Dividends"]:
        if column not in data.columns:
            data[column] = np.nan if column != "Dividends" else 0.0

    data["Dividends"] = data["Dividends"].fillna(0.0)
    return data


def annualized_return(prices: pd.Series) -> float:
    prices = prices.dropna()
    if len(prices) < 2 or prices.iloc[0] <= 0:
        return np.nan

    years = (prices.index[-1] - prices.index[0]).days / 365.25
    if years <= 0:
        return np.nan

    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1


def max_drawdown(prices: pd.Series) -> float:
    prices = prices.dropna()
    if prices.empty:
        return np.nan
    return (prices / prices.cummax() - 1).min()


def annualized_volatility(prices: pd.Series) -> float:
    returns = prices.dropna().pct_change().dropna()
    if len(returns) < 20:
        return np.nan
    return returns.std() * math.sqrt(252)


def recent_distribution_growth(dividends: pd.Series) -> float:
    """Compare latest 12 months of distributions with the preceding 12 months."""
    dividends = dividends.dropna()
    if dividends.empty:
        return np.nan

    end = dividends.index.max()
    latest_start = end - pd.DateOffset(years=1)
    prior_start = end - pd.DateOffset(years=2)

    latest = dividends.loc[dividends.index > latest_start].sum()
    prior = dividends.loc[
        (dividends.index > prior_start) & (dividends.index <= latest_start)
    ].sum()

    if prior <= 0:
        return np.nan
    return latest / prior - 1


def analyze_ticker(ticker: str, period: str) -> tuple[dict | None, str | None]:
    try:
        data = load_history(ticker, period)
        if data.empty or data["Close"].dropna().empty:
            return None, "No price history returned"

        latest_date = data["Close"].dropna().index[-1]
        latest_close = float(data.loc[latest_date, "Close"])

        one_year_start = latest_date - pd.DateOffset(years=1)
        ttm_distributions = float(
            data.loc[data.index > one_year_start, "Dividends"].sum()
        )
        trailing_yield = (
            ttm_distributions / latest_close if latest_close > 0 else np.nan
        )

        adjusted = data["Adj Close"].dropna()
        if adjusted.empty:
            adjusted = data["Close"].dropna()

        one_year_adjusted = adjusted.loc[adjusted.index > one_year_start]
        one_year_total_return = (
            one_year_adjusted.iloc[-1] / one_year_adjusted.iloc[0] - 1
            if len(one_year_adjusted) >= 2
            else np.nan
        )

        monthly_payments = (
            data.loc[data.index > one_year_start]
            .assign(Month=lambda x: x.index.to_period("M"))
            .groupby("Month")["Dividends"]
            .sum()
        )
        months_paid = int((monthly_payments > 0).sum())
        average_monthly_distribution = (
            ttm_distributions / months_paid if months_paid else 0.0
        )

        return {
            "Ticker": ticker,
            "Price": latest_close,
            "TTM Distribution": ttm_distributions,
            "TTM Yield": trailing_yield,
            "1Y Total Return": one_year_total_return,
            "Annualized Return": annualized_return(adjusted),
            "Volatility": annualized_volatility(adjusted),
            "Max Drawdown": max_drawdown(adjusted),
            "Distribution Growth": recent_distribution_growth(data["Dividends"]),
            "Months Paid": months_paid,
            "Avg Payment": average_monthly_distribution,
            "History Start": data.index.min().date(),
            "As Of": latest_date.date(),
        }, None

    except Exception as exc:
        return None, str(exc)


def calculate_score(
    frame: pd.DataFrame,
    yield_weight: float,
    return_weight: float,
    drawdown_weight: float,
    volatility_weight: float,
) -> pd.Series:
    """
    Percentile score:
      Higher yield and return are rewarded.
      Shallower drawdown and lower volatility are rewarded.
    """
    yield_rank = frame["TTM Yield"].rank(pct=True)
    return_rank = frame["1Y Total Return"].rank(pct=True)
    drawdown_rank = frame["Max Drawdown"].rank(pct=True)
    volatility_rank = (-frame["Volatility"]).rank(pct=True)

    total_weight = (
        yield_weight + return_weight + drawdown_weight + volatility_weight
    )
    if total_weight <= 0:
        return pd.Series(0.0, index=frame.index)

    score = (
        yield_rank * yield_weight
        + return_rank * return_weight
        + drawdown_rank * drawdown_weight
        + volatility_rank * volatility_weight
    ) / total_weight

    return score * 100


def portfolio_metrics(
    results: pd.DataFrame,
    weights: dict[str, float],
    investment: float,
) -> pd.DataFrame:
    rows = []
    total_weight = sum(weights.values())

    if total_weight <= 0:
        return pd.DataFrame()

    indexed = results.set_index("Ticker")

    for ticker, raw_weight in weights.items():
        if ticker not in indexed.index or raw_weight <= 0:
            continue

        weight = raw_weight / total_weight
        row = indexed.loc[ticker]
        allocation = investment * weight
        estimated_income = allocation * row["TTM Yield"]

        rows.append(
            {
                "Ticker": ticker,
                "Weight": weight,
                "Allocation": allocation,
                "TTM Yield": row["TTM Yield"],
                "Est. Annual Income": estimated_income,
                "Est. Monthly Income": estimated_income / 12,
                "1Y Total Return": row["1Y Total Return"],
                "Max Drawdown": row["Max Drawdown"],
            }
        )

    return pd.DataFrame(rows)


st.title("💵 Covered-Call ETF Scanner")
st.caption(
    "Find and compare high-distribution ETFs using trailing distributions, "
    "total return, volatility, and drawdown."
)

with st.sidebar:
    st.header("Scanner settings")

    ticker_text = st.text_area(
        "Candidate tickers",
        value=", ".join(DEFAULT_TICKERS),
        height=170,
        help="Enter comma-, space-, or line-separated symbols.",
    )

history_period=st.selectbox("History",options=["1y", "2y", "5y", "10y", "max"],index=1,)

minimum_yield = st.slider(
"Minimum trailing yield",
min_value=0.0,
max_value=30.0,
value=7.0,
step=0.5,
format="%.1f%%",
) / 100

    maximum_drawdown = st.slider(
        "Maximum tolerated drawdown",
        min_value=-80.0,
        max_value=0.0,
        value=-40.0,
        step=2.0,
        format="%.0f%%",
        help="ETFs with a historical drawdown worse than this are excluded.",
    ) / 100

    st.subheader("Ranking weights")
    yield_weight = st.slider("Yield", 0, 100, 35)
    return_weight = st.slider("1-year total return", 0, 100, 30)
    drawdown_weight = st.slider("Drawdown protection", 0, 100, 20)
    volatility_weight = st.slider("Lower volatility", 0, 100, 15)

    run_scan = st.button("Run ETF scan", type="primary", use_container_width=True)


if run_scan or "scan_results" not in st.session_state:
    tickers = normalize_tickers(ticker_text)

    rows = []
    errors = {}

    progress = st.progress(0, text="Downloading ETF history...")
    for index, ticker in enumerate(tickers):
        row, error = analyze_ticker(ticker, history_period)
        if row:
            rows.append(row)
        if error:
            errors[ticker] = error
        progress.progress(
            (index + 1) / max(len(tickers), 1),
            text=f"Analyzing {ticker}...",
        )
    progress.empty()

    st.session_state["scan_results"] = pd.DataFrame(rows)
    st.session_state["scan_errors"] = errors


results = st.session_state.get("scan_results", pd.DataFrame())
errors = st.session_state.get("scan_errors", {})

if results.empty:
    st.error("No ETF data was returned. Check the symbols and try again.")
    if errors:
        st.json(errors)
    st.stop()

results = results.copy()
results["Score"] = calculate_score(
    results,
    yield_weight,
    return_weight,
    drawdown_weight,
    volatility_weight,
)

filtered = results[
    (results["TTM Yield"] >= minimum_yield)
    & (results["Max Drawdown"] >= maximum_drawdown)
].sort_values(["Score", "TTM Yield"], ascending=False)

top1, top2, top3, top4 = st.columns(4)
top1.metric("ETFs analyzed", len(results))
top2.metric("Passing filters", len(filtered))
top3.metric(
    "Highest trailing yield",
    f"{results['TTM Yield'].max():.2%}",
)
top4.metric(
    "Best scanner score",
    f"{results['Score'].max():.1f}",
)

st.subheader("Ranked candidates")

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
    "As Of",
]

st.dataframe(
    filtered[display_columns],
    hide_index=True,
    use_container_width=True,
    column_config={
        "Score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=100, format="%.1f"
        ),
        "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
        "TTM Yield": st.column_config.NumberColumn("TTM Yield", format="%.2f%%"),
        "1Y Total Return": st.column_config.NumberColumn(
            "1Y Total Return", format="%.2f%%"
        ),
        "Annualized Return": st.column_config.NumberColumn(
            "Annualized Return", format="%.2f%%"
        ),
        "Max Drawdown": st.column_config.NumberColumn(
            "Max Drawdown", format="%.2f%%"
        ),
        "Volatility": st.column_config.NumberColumn(
            "Volatility", format="%.2f%%"
        ),
        "Distribution Growth": st.column_config.NumberColumn(
            "Distribution Growth", format="%.2f%%"
        ),
        "TTM Distribution": st.column_config.NumberColumn(
            "TTM Distribution", format="$%.2f"
        ),
    },
)

csv = filtered[display_columns].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download scan as CSV",
    data=csv,
    file_name=f"covered_call_etf_scan_{date.today().isoformat()}.csv",
    mime="text/csv",
)

if errors:
    with st.expander(f"Symbols with data errors ({len(errors)})"):
        st.json(errors)

st.divider()
st.header("Basket builder")

available_tickers = results.sort_values("Ticker")["Ticker"].tolist()
default_selected = [ticker for ticker in DEFAULT_BASKET if ticker in available_tickers]

selected = st.multiselect(
    "Choose ETFs",
    options=available_tickers,
    default=default_selected,
)

investment = st.number_input(
    "Portfolio amount",
    min_value=1_000.0,
    value=100_000.0,
    step=5_000.0,
    format="%.2f",
)

weights = {}
if selected:
    st.write("Set target weights:")
    weight_columns = st.columns(min(len(selected), 4))
    for index, ticker in enumerate(selected):
        default_weight = DEFAULT_BASKET.get(ticker, 100 / len(selected))
        with weight_columns[index % len(weight_columns)]:
            weights[ticker] = st.number_input(
                ticker,
                min_value=0.0,
                max_value=100.0,
                value=float(default_weight),
                step=5.0,
                key=f"weight_{ticker}",
                format="%.1f%%",
            )

basket = portfolio_metrics(results, weights, investment)

if not basket.empty:
    total_raw_weight = sum(weights.values())
    if not np.isclose(total_raw_weight, 100.0):
        st.info(
            f"Entered weights total {total_raw_weight:.1f}%. "
            "The app normalized them to 100%."
        )

    annual_income = basket["Est. Annual Income"].sum()
    monthly_income = basket["Est. Monthly Income"].sum()
    weighted_yield = annual_income / investment if investment else np.nan
    weighted_return = np.average(
        basket["1Y Total Return"].fillna(0),
        weights=basket["Weight"],
    )

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Portfolio yield", f"{weighted_yield:.2%}")
    metric2.metric("Estimated annual income", f"${annual_income:,.0f}")
    metric3.metric("Estimated monthly income", f"${monthly_income:,.0f}")
    metric4.metric("Weighted 1Y total return", f"{weighted_return:.2%}")

    st.dataframe(
        basket,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Weight": st.column_config.NumberColumn("Weight", format="%.1f%%"),
            "Allocation": st.column_config.NumberColumn(
                "Allocation", format="$%.2f"
            ),
            "TTM Yield": st.column_config.NumberColumn(
                "TTM Yield", format="%.2f%%"
            ),
            "Est. Annual Income": st.column_config.NumberColumn(
                "Est. Annual Income", format="$%.2f"
            ),
            "Est. Monthly Income": st.column_config.NumberColumn(
                "Est. Monthly Income", format="$%.2f"
            ),
            "1Y Total Return": st.column_config.NumberColumn(
                "1Y Total Return", format="%.2f%%"
            ),
            "Max Drawdown": st.column_config.NumberColumn(
                "Max Drawdown", format="%.2f%%"
            ),
        },
    )

    chart_data = basket.set_index("Ticker")[["Allocation"]]
    st.bar_chart(chart_data)

st.divider()
st.warning(
    "Trailing distribution yield is not the same as guaranteed investment "
    "income. Covered-call ETF distributions may include option premium, "
    "dividends, capital gains, and return of capital. Always review the "
    "issuer's distribution notices, tax documents, fees, strategy, and NAV trend."
)
