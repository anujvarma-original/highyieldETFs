# Covered-Call ETF Scanner

A Streamlit app that:

- scans a configurable list of covered-call and income ETFs;
- estimates trailing-12-month distribution yield;
- measures total return, volatility, maximum drawdown, and distribution growth;
- ranks ETFs with adjustable weights;
- builds a custom income basket;
- starts with a 30% QQQI / 30% JEPQ / 20% SPYI / 20% JEPI basket.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Put `app.py` and `requirements.txt` in a GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repository.
3. Set the main file path to `app.py`.
4. Deploy.

Market data comes from Yahoo Finance through the unofficial `yfinance` package.
The scanner is for research, not investment or tax advice.
