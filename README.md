Autonomous ML Portfolio Optimization & Trading Bot

An advanced, autonomous algorithmic trading and portfolio rebalancing system. It leverages **XGBoost** machine learning models for market direction prediction, **NLTK VADER** for real-time financial news sentiment analysis, and **Markowitz Efficient Frontier (Modern Portfolio Theory)** to optimize asset allocations daily.

All live trades are automatically executed via the **Alpaca API** (Paper Trading).

---

## Key Features

*   **Machine Learning (XGBoost):** Classifies and predicts the 5-day future price movement probability for assets based on technical indicators (RSI, MACD, MA crossovers, Volatility).
*   **NLP Sentiment Analysis:** Scrapes real-time financial news via Yahoo Finance and scores sentiment using NLTK's VADER engine.
*   **Modern Portfolio Theory (MPT):** Minimizes portfolio variance and maximizes the Sharpe Ratio dynamically using Scipy's optimization algorithms.
*   **Automated Execution:** Seamlessly connects to the Alpaca Brokerage API to fetch current equity, calculate target shares, and execute market orders (buys/sells) to rebalance the portfolio.
*   **Task Automation:** Fully compatible with Windows Task Scheduler / Cron Jobs for 100% hands-free daily execution.

---

## Managed Assets

The bot dynamically allocates capital among 4 diversified assets to manage risk:
1.  **SPY** (S&P 500 ETF - Equity)
2.  **QQQ** (Nasdaq 100 ETF - Tech Equity)
3.  **GLD** (Gold Trust ETF - Safe Haven Asset)
4.  **TLT** (20+ Year Treasury Bond ETF - Fixed Income)

---

## Tech Stack & Libraries

*   **Language:** Python 3.14+
*   **Data Retrieval:** `yfinance`
*   **Data Processing:** `pandas`, `numpy`, `scipy`
*   **Machine Learning:** `xgboost`, `scikit-learn`
*   **Natural Language Processing:** `nltk` (VADER sentiment)
*   **Brokerage Integration:** `requests` (Alpaca REST API v2)

---

## Setup and Installation

1. Clone this repository:
   ```bash
   git clone [https://github.com/kaan66tarikk-coder/ml-portfolio-rebalancing-bot.git](https://github.com/kaan66tarikk-coder/ml-portfolio-rebalancing-bot.git)
   ```

2. Install the required libraries:
   ```bash
   pip install yfinance pandas numpy xgboost nltk requests scipy
   ```

3. Open `daily_signal.py` and insert your Alpaca API credentials:
   ```python
   ALPACA_API_KEY = "YOUR_API_KEY"
   ALPACA_SECRET_KEY = "YOUR_SECRET_KEY"
   ```

4. Run the rebalancing system:
   ```bash
   python daily_signal.py
   ```

## Legal Disclaimer

This software is developed strictly for educational, research, and informational purposes. It does not constitute financial, investment, tax, or trading advice. Algorithmic trading and financial markets involve substantial risk of loss and are not suitable for every investor. The source code provided here is a personal project and is not intended for live trading with real capital. The developer assumes no liability or responsibility for any financial losses, damages, or decisions made based on the utilization of this code. Always test thoroughly using simulated environments (paper trading) before exposing actual capital to market risks.
