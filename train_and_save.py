import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import os
print("Downloading Gold data from Yahoo Finance...")
df = yf.download("GC=F", start="2020-01-01")
df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
df.columns = df.columns.str.lower()
print("Calculating technical indicators and sentiment features...")
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
price_momentum = df['daily_return'].rolling(3).mean()
rsi_scaled = (df['rsi'] - 50) / 50.0
df['sentiment_score'] = (price_momentum * 0.5 + rsi_scaled * 0.5).clip(-1, 1)
df_clean = df.dropna().copy()
features = ['close_to_ma_5', 'close_to_ma_50', 'ma_5_to_ma_50', 'daily_return', 'rsi', 'macd', 'macd_signal', 'volatility_10', 'relative_volume', 'sentiment_score']
X = df_clean[features]
y = df_clean['target']
print("Training XGBoost model with Sentiment features...")
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=1, eval_metric='logloss')
model.fit(X, y)
model_filename = "gold_xgb_model.json"
print(f"Saving model to local disk: {model_filename}")
model.save_model(model_filename)
print("\n" + "="*50)
print("SENTIMENT-ENABLED MODEL SUCCESSFULLY SAVED!")
print("="*50)
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import requests
ALPACA_API_KEY = ".........."
ALPACA_SECRET_KEY = "........."
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
        print(f"Warning: Could not calculate news sentiment ({e}). Using default neutral sentiment.")
        return 0.0
def get_alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json"
    }
def check_alpaca_position(symbol):
    url = f"{ALPACA_BASE_URL}/v2/positions/{symbol}"
    response = requests.get(url, headers=get_alpaca_headers())
    if response.status_code == 200:
        return True
    return False
def execute_alpaca_order(symbol, side, qty=1):
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
print("Fetching latest Gold market data...")
df = yf.download("GC=F", period="100d")
df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
df.columns = df.columns.str.lower()
print("Calculating live technical indicators...")
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
print("Analyzing live news headlines using NLTK NLP engine...")
live_sentiment = get_live_news_sentiment("GC=F")
df['sentiment_score'] = live_sentiment
df_clean = df.dropna().copy()
features = ['close_to_ma_5', 'close_to_ma_50', 'ma_5_to_ma_50', 'daily_return', 'rsi', 'macd', 'macd_signal', 'volatility_10', 'relative_volume', 'sentiment_score']
latest_features = df_clean[features].iloc[[-1]]
model_filename = "gold_xgb_model.json"
if not os.path.exists(model_filename):
    print(f"Error: Model file '{model_filename}' not found. Please run 'train_and_save.py' first.")
    exit()
model = XGBClassifier()
model.load_model(model_filename)
probability = model.predict_proba(latest_features)[0][1]
print("\n" + "="*50)
print("LIVE SIGNAL DECISION MATRIX")
print("="*50)
print(f"Analyzing Date: {df_clean.index[-1].strftime('%Y-%m-%d')}")
print(f"Calculated NLP News Sentiment Score: {live_sentiment:+.4f}")
print(f"Yol Gosterici Alim Ihtimalli (Probability): {probability*100:.2f}%")
trade_symbol = "GLD"
if probability > 0.65:
    print("FINAL RECOMMENDATION: BUY (Strong Bullish Sentiment and Technical Alignment)")
    if not check_alpaca_position(trade_symbol):
        print("No open position detected. Sending BUY order to Alpaca...")
        execute_alpaca_order(trade_symbol, "buy", qty=10)
    else:
        print("Position already exists. Holding.")
else:
    print("FINAL RECOMMENDATION: WAIT (Insufficient Market Momentum or Negative News Bias)")
    if check_alpaca_position(trade_symbol):
        print("Open position detected on WAIT signal. Closing position...")
        execute_alpaca_order(trade_symbol, "sell", qty=10)
    else:
        print("No open positions to manage.")
print("="*50)
