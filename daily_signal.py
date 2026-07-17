import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import requests
from scipy.optimize import minimize

ALPACA_API_KEY = "......"
ALPACA_SECRET_KEY = "......"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

nltk.download('vader_lexicon', quiet=True)

def get_live_news_sentiment(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news
        if not news:
            return 0.0
        sia = SentimentIntensityAnalyzer()
        scores = []
        for article in news[:10]:
            title = article.get('title', '')
            score = sia.polarity_scores(title)['compound']
            scores.append(score)
        return np.mean(scores) if scores else 0.0
    except Exception as e:
        print(f"Warning: News sentiment calculation failed: {e}")
        return 0.0

def get_alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json"
    }

def get_alpaca_positions():
    url = f"{ALPACA_BASE_URL}/v2/positions"
    response = requests.get(url, headers=get_alpaca_headers())
    if response.status_code == 200:
        return {pos["symbol"]: float(pos["qty"]) for pos in response.json()}
    return {}

def get_alpaca_equity():
    url = f"{ALPACA_BASE_URL}/v2/account"
    response = requests.get(url, headers=get_alpaca_headers())
    if response.status_code == 200:
        return float(response.json()["equity"])
    return 100000.0

def execute_alpaca_order(symbol, side, qty=1):
    if qty <= 0:
        return
    url = f"{ALPACA_BASE_URL}/v2/orders"
    data = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "day"
    }
    response = requests.post(url, json=data, headers=get_alpaca_headers())
    if response.status_code == 200 or response.status_code == 201:
        print(f"Alpaca Order Successful: {side.upper()} {qty} shares of {symbol}")
    else:
        print(f"Alpaca Order Failed: {response.text}")

def get_portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def objective_function(weights, expected_returns, cov_matrix, risk_free_rate=0.01):
    port_return = np.sum(expected_returns * weights)
    port_variance = get_portfolio_variance(weights, cov_matrix)
    port_std = np.sqrt(port_variance)
    sharpe_ratio = (port_return - risk_free_rate) / (port_std + 1e-8)
    return -sharpe_ratio

def train_and_predict_asset(symbol):
    df = yf.download(symbol, start="2020-01-01")
    df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df.columns = df.columns.str.lower()
    df['future_close'] = df['close'].shift(-5)
    df['target'] = (df['future_close'] > df['close']).astype(int)
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_50'] = df['close'].rolling(50).mean()
    df['close_to_ma_5'] = df['close'] / df['ma_5']
    df['close_to_ma_50'] = df['close'] / df['ma_50']
    df['ma_5_to_ma_50'] = df['ma_5'] / df['ma_50']
    df['daily_return'] = df['close'].pct_change()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['volatility_10'] = df['daily_return'].rolling(10).std()
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['relative_volume'] = df['volume'] / df['volume_ma_5']
    live_sentiment = get_live_news_sentiment(symbol)
    df['sentiment_score'] = live_sentiment
    df_clean = df.dropna().copy()
    features = ['close_to_ma_5', 'close_to_ma_50', 'ma_5_to_ma_50', 'daily_return', 'rsi', 'macd', 'macd_signal', 'volatility_10', 'relative_volume', 'sentiment_score']
    X = df_clean[features]
    y = df_clean['target']
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=1, eval_metric='logloss')
    model.fit(X, y)
    latest_features = df_clean[features].iloc[[-1]]
    probability = model.predict_proba(latest_features)[0][1]
    historical_returns = df_clean['daily_return'].tail(100)
    return probability, historical_returns, live_sentiment, df_clean['close'].iloc[-1]

def run_rebalancing_system():
    assets = ["GLD", "SPY", "QQQ", "TLT"]
    expected_returns = []
    returns_dict = {}
    live_sentiments = {}
    last_prices = {}
    for asset in assets:
        print(f"Processing and predicting {asset}...")
        prob, hist_ret, sentiment, last_p = train_and_predict_asset(asset)
        expected_returns.append(prob)
        returns_dict[asset] = hist_ret
        live_sentiments[asset] = sentiment
        last_prices[asset] = last_p
    expected_returns = np.array(expected_returns)
    returns_df = pd.DataFrame(returns_dict).dropna()
    cov_matrix = returns_df.cov().values * 252
    num_assets = len(assets)
    init_weights = np.array([1.0 / num_assets] * num_assets)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    optimized = minimize(objective_function, init_weights, args=(expected_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = optimized.x
    equity = get_alpaca_equity()
    current_positions = get_alpaca_positions()
    print("\n" + "="*50)
    print("PORTFOLIO OPTIMIZATION AND REBALANCING MATRIX")
    print("="*50)
    print(f"Total Portfolio Equity: ${equity:.2f}")
    orders_to_execute = []
    for i, asset in enumerate(assets):
        target_weight = optimal_weights[i]
        target_value = equity * target_weight
        current_price = last_prices[asset]
        target_qty = int(target_value / current_price)
        current_qty = current_positions.get(asset, 0.0)
        qty_diff = target_qty - current_qty
        orders_to_execute.append((asset, qty_diff, target_weight, target_qty))
        print(f"Asset: {asset} | Sentiment: {live_sentiments[asset]:+.4f} | Optimal Weight: {target_weight*100:.2f}% | Target Qty: {target_qty} | Current Qty: {int(current_qty)}")
    print("\nExecuting Sells First...")
    for asset, qty_diff, _, _ in orders_to_execute:
        if qty_diff < 0:
            execute_alpaca_order(asset, "sell", int(abs(qty_diff)))
    print("\nExecuting Buys Next...")
    for asset, qty_diff, _, _ in orders_to_execute:
        if qty_diff > 0:
            execute_alpaca_order(asset, "buy", int(qty_diff))
    print("="*50)

if __name__ == "__main__":
    run_rebalancing_system()
