# High-Income ETF Research Platform

## Files

- `app.py` — Streamlit user interface
- `analytics.py` — calculations, scoring, forecasting and portfolio search
- `etf_catalog.py` — ETF lookup metadata
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

Push all four source files to the same GitHub repository. Set `app.py` as the main file.

## Important limitations

Yahoo Finance does not consistently supply the tax character of ETF distributions. The application therefore does not claim that a specific payment is or is not return of capital. Check the issuer's Section 19 notices and annual tax statements.

The distribution forecast is a recency-weighted historical estimate, not a prediction based on current option-chain implied volatility.
