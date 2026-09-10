import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,accuracy_score, ConfusionMatrixDisplay)

plt.style.use('seaborn-v0_8-whitegrid')


# ── Config ──────────────────────────────────────────────────────────────────
WINDOW          = 256      # training window size (trading days)
RETRAIN_EVERY   = 10       # retrain cadence in the walk-forward loop
VOL_LOOKBACK    = 20       # window for volatility-scaled threshold
THRESHOLD_K     = 0.75     # threshold = K * trailing return std (tune this)
TXN_COST_BPS    = 5        # round-trip cost in basis points, per trade

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=0,
    n_jobs=-1,
)


# ── Load & clean data ──────────────────────────────────────────────────────────
df = pd.read_csv('APPLE_daily_reformed.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

n_before = len(df)
for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna().reset_index(drop=True)
n_after = len(df)
print(f"Rows : {n_after:,}  (dropped {n_before - n_after:,} rows with bad/missing values)")
print(f"Range: {df['Date'].min().date()} -> {df['Date'].max().date()}")

df['Price'] = df['Adj Close']


# ── Feature engineering ────────────────────────────────────────────────────────
def add_features(df):
    d = df.copy()
    p = d['Price']  # adjusted price, used for all return/trend/vol features

    d['Return_1d'] = p.pct_change()
    d['Return_2d'] = p.pct_change(2)
    d['Return_5d'] = p.pct_change(5)
    d['Return_10d'] = p.pct_change(10)
    d['Return_20d'] = p.pct_change(20)

    d['MA_5'] = p.rolling(5).mean()
    d['MA_20'] = p.rolling(20).mean()
    d['MA_50'] = p.rolling(50).mean()
    d['Price_vs_MA5']  = p / d['MA_5']  - 1
    d['Price_vs_MA20'] = p / d['MA_20'] - 1
    d['Price_vs_MA50'] = p / d['MA_50'] - 1
    d['MA5_vs_MA20'] = d['MA_5']  / d['MA_20'] - 1

    delta = p.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['RSI'] = 100 - 100 / (1 + gain / loss)

    bb_mid = p.rolling(20).mean()
    bb_std = p.rolling(20).std()
    d['BB_position'] = (p - (bb_mid - 2 * bb_std)) / (4 * bb_std)
    d['BB_width']    = (4 * bb_std) / bb_mid

    d['Volatility_10']  = d['Return_1d'].rolling(10).std()
    d['Volatility_20']  = d['Return_1d'].rolling(VOL_LOOKBACK).std()
    d['Volume_ratio']   = d['Volume'] / d['Volume'].rolling(20).mean()

    d['High_Low_pct']   = (d['High'] - d['Low']) / d['Close']
    d['Open_Close_pct'] = (d['Close'] - d['Open']) / d['Open']

    d['Lag1_return'] = d['Return_1d'].shift(1)
    d['Lag2_return'] = d['Return_1d'].shift(2)
    d['Lag5_return'] = d['Return_1d'].shift(5)
    return d


features = [
    'Return_1d', 'Return_2d', 'Return_5d', 'Return_10d', 'Return_20d',
    'Price_vs_MA5', 'Price_vs_MA20', 'Price_vs_MA50', 'MA5_vs_MA20',
    'RSI', 'BB_position', 'BB_width',
    'Volatility_10', 'Volatility_20', 'Volume_ratio',
    'High_Low_pct', 'Open_Close_pct',
    'Lag1_return', 'Lag2_return', 'Lag5_return',
]

df = add_features(df)

df['Next_Return'] = df['Price'].pct_change().shift(-1)
df['Vol_threshold'] = THRESHOLD_K * df['Return_1d'].rolling(VOL_LOOKBACK).std()

def label_signal(row):
    thr = row['Vol_threshold']
    ret = row['Next_Return']
    if pd.isna(thr) or pd.isna(ret):
        return np.nan
    if ret > thr:
        return 'Buy'
    elif ret < -thr:
        return 'Sell'
    else:
        return 'Hold'

df['Signal'] = df.apply(label_signal, axis=1)
df = df.dropna(subset=features + ['Signal']).reset_index(drop=True)

print(f'\n{len(df):,} usable rows | {len(features)} features')
print(df['Signal'].value_counts())
print(f'First prediction date: {df["Date"].iloc[WINDOW].date()}')


# ── Walk-forward evaluation ────────────────────────────────────────────────────
X      = df[features].values
y      = df['Signal'].values
dates  = df['Date'].values
prices = df['Price'].values

results = []
clf     = None

print(f"\nWalk-forward: {len(df)-WINDOW:,} predictions to make...")

for i in range(WINDOW, len(df)):
    step = i - WINDOW

    if step % RETRAIN_EVERY == 0:
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X[i - WINDOW:i], y[i - WINDOW:i])

        if step % 500 == 0:
            print(f"  Step {step:,}/{len(df)-WINDOW:,}  "
                  f"({step / (len(df)-WINDOW) * 100:.1f}%)")

    pred_signal = clf.predict(X[i].reshape(1, -1))[0]
    pred_proba  = clf.predict_proba(X[i].reshape(1, -1))[0]
    classes     = clf.classes_

    results.append({
        'Date':      pd.Timestamp(dates[i]),
        'Price':     prices[i],
        'Actual':    y[i],
        'Predicted': pred_signal,
        **{f'P_{c}': p for c, p in zip(classes, pred_proba)},
    })

res = pd.DataFrame(results)
print("\nDone!")
print(res.tail(8).to_string())


# ── Classification metrics ─────────────────────────────────────────────────────
acc = accuracy_score(res['Actual'], res['Predicted'])
print(f"\nOverall Accuracy: {acc * 100:.2f}%\n")
print(classification_report(res['Actual'], res['Predicted'],
                             target_names=['Buy', 'Hold', 'Sell']))

for sig in ['Buy', 'Sell']:
    predicted_mask = res['Predicted'] == sig
    if predicted_mask.sum() > 0:
        precision = (res.loc[predicted_mask, 'Actual'] == sig).mean()
        print(f"{sig} precision: {precision*100:.1f}%  "
              f"({predicted_mask.sum():,} signals issued)")

hold_frac = (res['Actual'] == 'Hold').mean()
print(f"\nMajority-class (always-Hold) baseline accuracy: {hold_frac*100:.2f}%")


# ── Confusion matrix ───────────────────────────────────────────────────────────
cm   = confusion_matrix(res['Actual'], res['Predicted'], labels=['Buy', 'Hold', 'Sell'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Buy', 'Hold', 'Sell'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap='Greens')
ax.set_title('Random Forest Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()


# ── Signal chart ───────────────────────────────────────────────────────────────
color_map = {'Buy': '#2ecc71', 'Sell': '#e74c3c', 'Hold': '#95a5a6'}

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(res['Date'], res['Price'], color='#4fc3f7',
        linewidth=1.5, label='Adj Close', zorder=1)

for signal, color in color_map.items():
    mask = res['Predicted'] == signal
    ax.scatter(res.loc[mask, 'Date'], res.loc[mask, 'Price'], color=color, s=12, label=signal, zorder=2, alpha=0.7)

ax.set_title('Random Forest — Buy / Sell / Hold Signals on Price', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Adjusted Close (USD)')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('signals_on_price.png', dpi=150)
plt.show()


# ── Backtest: does trading the signal actually make money? ────────────────────

res = res.sort_values('Date').reset_index(drop=True)
res['Day_return'] = res['Price'].pct_change().shift(-1)  # realized t -> t+1 return

position = (res['Predicted'] == 'Buy').astype(int)
trades = position.diff().abs().fillna(0)  # 1 whenever position flips
txn_cost = trades * (TXN_COST_BPS / 10_000)

strategy_return = position * res['Day_return'] - txn_cost
buy_and_hold_ret  = res['Day_return']

res['Strategy_equity'] = (1 + strategy_return.fillna(0)).cumprod()
res['BuyHold_equity']  = (1 + buy_and_hold_ret.fillna(0)).cumprod()

total_strategy_return = res['Strategy_equity'].iloc[-1] - 1
total_bh_return        = res['BuyHold_equity'].iloc[-1] - 1

sharpe = (strategy_return.mean() / strategy_return.std()) * np.sqrt(252) \
         if strategy_return.std() > 0 else np.nan

print("\n" + "=" * 50)
print("  BACKTEST (long-only, trades on Buy signal only)")
print("=" * 50)
print(f"  Strategy total return : {total_strategy_return*100:8.2f}%")
print(f"  Buy & hold total return: {total_bh_return*100:8.2f}%")
print(f"  Strategy Sharpe (ann.) : {sharpe:8.2f}")
print(f"  Number of trades       : {int(trades.sum()):,}")
print(f"  Txn cost assumed       : {TXN_COST_BPS} bps per flip")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(res['Date'], res['Strategy_equity'], label='Strategy', color='#2ecc71')
ax.plot(res['Date'], res['BuyHold_equity'], label='Buy & Hold', color='#95a5a6')
ax.set_title('Equity Curve: Strategy vs Buy & Hold', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Growth of $1')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('equity_curve.png', dpi=150)
plt.show()


# ── Next-day prediction ────────────────────────────────────────────────────────
X_final = df[features].values[-WINDOW:]
y_final = df['Signal'].values[-WINDOW:]

clf_final = RandomForestClassifier(**RF_PARAMS)  # same params as walk-forward loop
clf_final.fit(X_final, y_final)

latest = df[features].iloc[[-1]]
pred_signal = clf_final.predict(latest)[0]
pred_proba = clf_final.predict_proba(latest)[0]
classes = clf_final.classes_
last_date = df['Date'].iloc[-1].date()
next_date = last_date + pd.Timedelta(days=1)

print("=" * 50)
print(f"  NEXT DAY SIGNAL  (from {last_date})")
print(f"  Predicting for:  {next_date}")
print("=" * 50)
print(f"  Signal: {pred_signal}")
print()
print("  Probabilities:")
for c, p in zip(classes, pred_proba):
    print(f"    {c:4s}: {p * 100:.2f}%")