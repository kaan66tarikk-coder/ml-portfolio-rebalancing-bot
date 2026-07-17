import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score
df = yf.download("GC=F", start="2020-01-01", end="2026-07-16")
df.columns = df.columns.droplevel(1)
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
df_clean = df.dropna().copy()
features = ['close_to_ma_5', 'close_to_ma_50', 'ma_5_to_ma_50', 'daily_return', 'rsi', 'macd', 'macd_signal', 'volatility_10', 'relative_volume']
X = df_clean[features]
y = df_clean['target']
split_index = int(len(df_clean) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=1, eval_metric='logloss')
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:, 1]
probabilities = pd.Series(probabilities, index=y_test.index)
custom_predictions = (probabilities > 0.65).astype(int)
test_data = df_clean.iloc[split_index:].copy()
test_data['pred'] = custom_predictions
test_data['next_5d_return'] = (test_data['future_close'] - test_data['close']) / test_data['close']
test_data['strategy_return'] = np.where(test_data['pred'] == 1, test_data['next_5d_return'], 0)
total_strategy_return = test_data['strategy_return'].sum() * 100
buy_and_hold = ((test_data['close'].iloc[-1] - test_data['close'].iloc[0]) / test_data['close'].iloc[0]) * 100
last_row = df.tail(1)
last_features = last_row[features]
live_prob = model.predict_proba(last_features)[:, 1][0]
print("--- XGBOOST BACKTEST RESULTS (THRESHOLD 0.65) ---")
print("Success Rate (%):", precision_score(y_test, custom_predictions) * 100)
print("Total Buy Signals:", custom_predictions.sum())
print("Strategy Total Return (%):", total_strategy_return)
print("Buy & Hold Gold Return (%):", buy_and_hold)
print("--- LIVE TODAY'S SIGNAL ---")
print("Today's Gold Close Price:", last_row['close'].values[0])
print("Model's Upward Probability for next 5 days:", live_prob * 100)
if live_prob > 0.65:
    print("SIGNAL: BUY! (Model predicts UP with high confidence)")
else:
    print("SIGNAL: WAIT / NO SIGNAL (Confidence below 65%)")
