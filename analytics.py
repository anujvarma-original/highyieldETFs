from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252


def safe_float(value: Any) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def download_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    history = yf.Ticker(ticker).history(
        period=period,
        auto_adjust=False,
        actions=True,
    )
    if history is None or history.empty:
        return pd.DataFrame()
    history = history.copy()
    if isinstance(history.index, pd.DatetimeIndex):
        try:
            history.index = history.index.tz_localize(None)
        except TypeError:
            pass
    return history.sort_index()


def price_series(history: pd.DataFrame) -> pd.Series:
    for column in ("Adj Close", "Close"):
        if column in history.columns:
            series = pd.to_numeric(history[column], errors="coerce").dropna()
            if not series.empty:
                return series
    return pd.Series(dtype=float)


def dividend_series(history: pd.DataFrame) -> pd.Series:
    if "Dividends" not in history.columns:
        return pd.Series(dtype=float)
    result = pd.to_numeric(history["Dividends"], errors="coerce").fillna(0)
    return result[result > 0]


def total_return_index(history: pd.DataFrame) -> pd.Series:
    prices = price_series(history)
    if prices.empty:
        return prices
    dividends = dividend_series(history).reindex(prices.index, fill_value=0.0)
    daily_total_return = prices.pct_change().fillna(0) + dividends / prices.shift(1)
    return (1 + daily_total_return.fillna(0)).cumprod()


def cagr(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) < 2 or series.iloc[0] <= 0:
        return np.nan
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return safe_float((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1)


def max_drawdown(series: pd.Series) -> float:
    series = series.dropna()
    if series.empty:
        return np.nan
    return safe_float((series / series.cummax() - 1).min())


def ulcer_index(series: pd.Series) -> float:
    series = series.dropna()
    if series.empty:
        return np.nan
    drawdowns_pct = (series / series.cummax() - 1) * 100
    return safe_float(np.sqrt(np.mean(np.square(drawdowns_pct))))


def annualized_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    return safe_float(returns.std() * np.sqrt(TRADING_DAYS)) if len(returns) > 2 else np.nan


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    returns = returns.dropna()
    if len(returns) < 3 or returns.std() == 0:
        return np.nan
    excess_daily = returns - risk_free_rate / TRADING_DAYS
    return safe_float(excess_daily.mean() / returns.std() * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    returns = returns.dropna()
    downside = returns[returns < 0]
    if len(returns) < 3 or len(downside) < 2 or downside.std() == 0:
        return np.nan
    excess_daily = returns.mean() - risk_free_rate / TRADING_DAYS
    return safe_float(excess_daily / downside.std() * np.sqrt(TRADING_DAYS))


def beta_and_correlation(returns: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    if len(aligned) < 20:
        return np.nan, np.nan
    aligned.columns = ["asset", "benchmark"]
    variance = aligned["benchmark"].var()
    beta = aligned["asset"].cov(aligned["benchmark"]) / variance if variance else np.nan
    correlation = aligned["asset"].corr(aligned["benchmark"])
    return safe_float(beta), safe_float(correlation)


def distribution_stability(dividends: pd.Series, expected_frequency: str) -> tuple[float, dict[str, float]]:
    if dividends.empty:
        return np.nan, {}
    monthly = dividends.resample("ME").sum()
    monthly = monthly.loc[monthly.index >= monthly.index.max() - pd.DateOffset(months=24)]
    if len(monthly) < 3:
        return np.nan, {}

    positive = monthly[monthly > 0]
    mean = positive.mean() if not positive.empty else 0
    cv = positive.std() / mean if mean > 0 and len(positive) > 1 else 1.0
    cuts = int((positive.pct_change() < -0.10).sum()) if len(positive) > 1 else 0
    skipped = int((monthly <= 0).sum()) if expected_frequency.lower() == "monthly" else 0

    recent_12 = monthly.tail(12).sum()
    previous_12 = monthly.iloc[-24:-12].sum() if len(monthly) >= 24 else np.nan
    growth = recent_12 / previous_12 - 1 if pd.notna(previous_12) and previous_12 > 0 else 0

    score = 100.0
    score -= min(40, cv * 45)
    score -= min(25, cuts * 5)
    score -= min(20, skipped * 5)
    score += max(-10, min(10, growth * 20))
    score = float(np.clip(score, 0, 100))
    return score, {"coefficient_of_variation": cv, "cuts": cuts, "skipped_months": skipped, "growth": growth}


def forecast_next_distribution(dividends: pd.Series) -> tuple[float, str]:
    """Simple historical estimate, not an options-pricing model."""
    if dividends.empty:
        return np.nan, "Unavailable"
    recent = dividends.tail(12)
    if len(recent) < 3:
        return safe_float(recent.mean()), "Low confidence"
    weights = np.arange(1, len(recent) + 1, dtype=float)
    estimate = np.average(recent.values, weights=weights)
    cv = recent.std() / recent.mean() if recent.mean() > 0 else np.inf
    confidence = "Higher" if len(recent) >= 10 and cv < 0.15 else "Moderate" if cv < 0.35 else "Low"
    return safe_float(estimate), confidence


def analyze_ticker(
    ticker: str,
    metadata: dict[str, Any],
    investment: float,
    period: str,
    spy_returns: pd.Series,
    qqq_returns: pd.Series,
    risk_free_rate: float,
) -> tuple[dict[str, Any] | None, pd.DataFrame, pd.DataFrame, str | None]:
    try:
        history = download_history(ticker, period)
        prices = price_series(history)
        if prices.empty:
            return None, pd.DataFrame(), pd.DataFrame(), f"No price data for {ticker}"

        dividends = dividend_series(history)
        total_index = total_return_index(history)
        returns = total_index.pct_change().dropna()
        current_price = safe_float(prices.iloc[-1])
        shares = investment / current_price

        cutoff = prices.index.max() - pd.Timedelta(days=365)
        ttm_dividend = safe_float(dividends.loc[dividends.index > cutoff].sum()) if not dividends.empty else 0.0
        annual_income = shares * ttm_dividend
        monthly_income = annual_income / 12
        trailing_yield = ttm_dividend / current_price if current_price > 0 else np.nan

        monthly = dividends.resample("ME").sum() if not dividends.empty else pd.Series(dtype=float)
        monthly_table = pd.DataFrame({
            "Month": monthly.index,
            "Distribution Per Share": monthly.values,
            "Position Income": monthly.values * shares,
            "Ticker": ticker,
        })

        forecast, forecast_confidence = forecast_next_distribution(dividends)
        stability, stability_detail = distribution_stability(dividends, metadata.get("frequency", "Monthly"))
        spy_beta, spy_corr = beta_and_correlation(returns, spy_returns)
        _, qqq_corr = beta_and_correlation(returns, qqq_returns)

        row = {
            "Ticker": ticker,
            "ETF Name": metadata.get("name", ticker),
            "Issuer": metadata.get("issuer", "Unknown"),
            "Strategy": metadata.get("strategy", "Unknown"),
            "Exposure": metadata.get("exposure", "Unknown"),
            "Distribution Frequency": metadata.get("frequency", "Unknown"),
            "Risk Band": metadata.get("risk_band", "Unknown"),
            "Expense Ratio": metadata.get("expense_ratio", np.nan),
            "ROC Note": metadata.get("roc_note", "Verify fund tax documents."),
            "Current Price": current_price,
            "Investment": investment,
            "Shares": shares,
            "TTM Distribution/Share": ttm_dividend,
            "TTM Distribution Yield": trailing_yield,
            "Estimated Monthly Income": monthly_income,
            "Estimated Annual Income": annual_income,
            "Forecast Next Distribution/Share": forecast,
            "Forecast Position Payment": forecast * shares if pd.notna(forecast) else np.nan,
            "Forecast Confidence": forecast_confidence,
            "Income Stability Score": stability,
            "Distribution Cuts (24M)": stability_detail.get("cuts", np.nan),
            "Price CAGR": cagr(prices),
            "Total Return CAGR": cagr(total_index),
            "Maximum Drawdown": max_drawdown(total_index),
            "Ulcer Index": ulcer_index(total_index),
            "Annualized Volatility": annualized_volatility(returns),
            "Sharpe Ratio": sharpe_ratio(returns, risk_free_rate),
            "Sortino Ratio": sortino_ratio(returns, risk_free_rate),
            "Beta vs SPY": spy_beta,
            "Correlation vs SPY": spy_corr,
            "Correlation vs QQQ": qqq_corr,
        }
        price_frame = pd.DataFrame({"Date": total_index.index, "Total Return Index": total_index.values, "Ticker": ticker})
        return row, monthly_table, price_frame, None
    except Exception as exc:
        return None, pd.DataFrame(), pd.DataFrame(), f"{ticker}: {type(exc).__name__}: {exc}"


def random_portfolio_search(
    daily_returns: pd.DataFrame,
    annual_yields: pd.Series,
    annual_incomes_per_dollar: pd.Series,
    simulations: int,
    max_weight: float,
    objective: str,
    risk_free_rate: float,
    seed: int = 42,
) -> pd.DataFrame:
    assets = daily_returns.columns.tolist()
    if len(assets) < 2:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(simulations):
        raw = rng.random(len(assets))
        weights = raw / raw.sum()
        if weights.max() > max_weight:
            continue
        portfolio_daily = daily_returns.fillna(0).values @ weights
        annual_return = (1 + pd.Series(portfolio_daily)).prod() ** (TRADING_DAYS / max(len(portfolio_daily), 1)) - 1
        volatility = np.std(portfolio_daily, ddof=1) * np.sqrt(TRADING_DAYS)
        sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else np.nan
        income_yield = float(np.dot(weights, annual_incomes_per_dollar.reindex(assets).fillna(0).values))
        score = {
            "Maximum income": income_yield,
            "Best Sharpe": sharpe,
            "Lowest volatility": -volatility,
            "Balanced": income_yield + max(sharpe, -5) * 0.02,
        }.get(objective, sharpe)
        records.append({
            **{f"Weight {asset}": weight for asset, weight in zip(assets, weights)},
            "Income Yield": income_yield,
            "Annualized Return": annual_return,
            "Volatility": volatility,
            "Sharpe": sharpe,
            "Score": score,
        })
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("Score", ascending=False)


def project_income(
    starting_value: float,
    annual_yield: float,
    annual_total_return: float,
    monthly_contribution: float,
    years: int,
    reinvest: bool,
) -> pd.DataFrame:
    balance = starting_value
    contributed = starting_value
    records = []
    monthly_yield = annual_yield / 12
    monthly_growth_ex_income = (1 + max(annual_total_return - annual_yield, -0.99)) ** (1 / 12) - 1
    for month in range(1, years * 12 + 1):
        income = balance * monthly_yield
        balance *= 1 + monthly_growth_ex_income
        if reinvest:
            balance += income
        balance += monthly_contribution
        contributed += monthly_contribution
        if month % 12 == 0:
            records.append({
                "Year": month // 12,
                "Portfolio Value": balance,
                "Annual Income at Current Yield": balance * annual_yield,
                "Monthly Income at Current Yield": balance * annual_yield / 12,
                "Cumulative Contributions": contributed,
                "Yield on Original Capital": (balance * annual_yield / starting_value) if starting_value else np.nan,
            })
    return pd.DataFrame(records)
