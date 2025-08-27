"""
donchian_backtest_report.py

Requirements: pandas, numpy, matplotlib, scipy
Install via: pip install pandas numpy matplotlib scipy
"""

import json
import math
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean

# -------------------------
# Utility / Metrics helpers
# -------------------------
def periods_per_year_from_timeframe(tf: str) -> float:
    """Approximate periods per year based on timeframe string like '1h','15m','1d'"""
    tf = tf.lower().strip()
    if tf.endswith('m'):  # minutes
        minutes = int(tf[:-1])
        return (365 * 24 * 60) / minutes
    if tf.endswith('h'):
        hours = int(tf[:-1])
        return (365 * 24) / hours
    if tf.endswith('d'):
        days = int(tf[:-1])
        return 365 / days
    if tf == '1w' or tf.endswith('w'):
        return 52
    # fallback to daily
    return 365

def max_drawdown(equity_curve: pd.Series):
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    md = drawdown.min()
    # find drawdown period
    end = drawdown.idxmin()
    start = equity_curve[:end].idxmax() if end is not None else None
    return md, start, end

def sharpe_ratio(returns, periods_per_year, rf=0.0):
    # returns: per-period returns (decimal)
    ann_ret = (1 + returns.mean()) ** periods_per_year - 1
    ann_vol = returns.std() * math.sqrt(periods_per_year)
    if ann_vol == 0:
        return np.nan
    return (ann_ret - rf) / ann_vol

def sortino_ratio(returns, periods_per_year, rf=0.0):
    # downside deviation
    neg_returns = returns[returns < 0]
    if len(neg_returns) == 0:
        return np.nan
    dd = neg_returns.std() * math.sqrt(periods_per_year)
    ann_ret = (1 + returns.mean()) ** periods_per_year - 1
    return (ann_ret - rf) / dd if dd != 0 else np.nan

# -------------------------
# Donchian breakout
# -------------------------
def donchian_breakout(df: pd.DataFrame, lookback: int):
    """
    upper = rolling max of close over (lookback-1) then shift(1) to avoid lookahead
    lower = rolling min of close over (lookback-1) then shift(1)
    signal: 1 if close > upper, -1 if close < lower, otherwise carry forward last signal (ffill)
    """
    df = df.copy()
    df['upper'] = df['close'].rolling(lookback - 1, min_periods=1).max().shift(1)
    df['lower'] = df['close'].rolling(lookback - 1, min_periods=1).min().shift(1)
    df['signal'] = np.nan
    df.loc[df['close'] > df['upper'], 'signal'] = 1
    df.loc[df['close'] < df['lower'], 'signal'] = -1
    df['signal'] = df['signal'].ffill().fillna(0)  # start neutral as 0
    return df

# -------------------------
# Trade simulator
# -------------------------
def simulate_trades(df: pd.DataFrame, config: dict):
    """
    Simulate trades based on df['signal'].
    - entry when signal changes to 1 or -1 (from previous not same)
    - exit when signal flips to opposite sign
    Costs:
      - commission_pct applied to notional on entry and exit
      - spread_pct applied as immediate adverse price (half-spread on entry & half on exit by sign)
      - slippage_pct applied adverse (as fraction of price) on entry and exit
    Position sizing:
      - notional = equity_at_entry * position_size_fraction * leverage
    Returns trade log DataFrame, equity_series (indexed by original timestamps), and trades summary
    """
    commission_pct = config['execution'].get('commission_pct', 0.0005)  # e.g., 0.05%
    slippage_pct = config['execution'].get('slippage_pct', 0.0005)      # 0.05% slippage per side
    spread_pct = config['execution'].get('spread_pct', 0.0002)          # e.g., 0.02% round-trip => half each side
    start_capital = config['execution'].get('start_capital', 10000.0)
    pos_fraction = config['execution'].get('position_size_fraction', 1.0) # fraction of equity per trade
    leverage = config['execution'].get('leverage', 1.0)
    timeframe = config.get('timeframe', '1h')
    periods_per_year = periods_per_year_from_timeframe(timeframe)

    equity = start_capital
    equity_hist = []
    equity_index = []
    trades = []

    in_trade = False
    entry_idx = None
    entry_price = None
    entry_signal = 0
    entry_equity = equity
    notional = 0.0

    # iterate rows
    for idx, row in df.iterrows():
        price = row['close']
        sig = int(row['signal'])
        ts = row['timestamp']

        # check for entry
        if not in_trade:
            # Enter when signal is 1 or -1 (non-zero)
            if sig != 0:
                in_trade = True
                entry_idx = idx
                entry_signal = sig
                entry_equity = equity
                notional = entry_equity * pos_fraction * leverage

                # apply spread & slippage adverse to entry price:
                # if long (1): pay ask = price * (1 + spread/2 + slippage)
                # if short (-1): sell at bid = price * (1 - spread/2 - slippage)
                entry_price = price * (1 + (spread_pct / 2.0 + slippage_pct) * entry_signal)
                # Actually for short sign(-1) multiplication by (1 + x * -1) works as (1 - x)
                # But to avoid confusion compute explicitly:
                if entry_signal == 1:
                    entry_price = price * (1 + spread_pct / 2.0 + slippage_pct)
                else:
                    entry_price = price * (1 - spread_pct / 2.0 - slippage_pct)

                entry_commission = commission_pct * notional  # commission charged on notional at entry
                # record a provisional trade dictionary; exit fields will be filled later
                trade = {
                    'entry_index': entry_idx,
                    'entry_timestamp': ts,
                    'entry_price': entry_price,
                    'entry_signal': entry_signal,
                    'entry_equity': entry_equity,
                    'notional': notional,
                    'entry_commission': entry_commission,
                    'exit_index': None,
                    'exit_timestamp': None,
                    'exit_price': None,
                    'exit_commission': None,
                    'gross_pnl': None,
                    'net_pnl': None,
                    'return_pct': None,
                    'duration': None
                }
                trades.append(trade)

        else:
            # in trade: check for exit condition => signal flips to opposite sign or zero
            if sig != entry_signal:
                # close trade at current row price with spread & slippage adverse for opposite side
                if sig == 0:
                    # exit to neutral, exit price determined by current price (apply spread/slippage conservative)
                    if entry_signal == 1:
                        exit_price = price * (1 - (spread_pct / 2.0 + slippage_pct))
                    else:
                        exit_price = price * (1 + (spread_pct / 2.0 + slippage_pct))
                else:
                    # flip to opposite side -> close current trade and we will immediately open a new one on next loop iteration or same row?
                    # We'll close this trade now at price with adverse spread/slippage for the exit side.
                    if entry_signal == 1:
                        # exiting long => selling, pay half-spread on exit (bid) and slippage adverse
                        exit_price = price * (1 - (spread_pct / 2.0 + slippage_pct))
                    else:
                        # exiting short => buy back, pay ask
                        exit_price = price * (1 + (spread_pct / 2.0 + slippage_pct))

                exit_commission = commission_pct * notional

                # For PnL:
                # long pnl = (exit_price - entry_price) / entry_price * notional
                # short pnl = (entry_price - exit_price) / entry_price * notional
                if entry_signal == 1:
                    gross_pnl = (exit_price - entry_price) / entry_price * notional
                else:
                    gross_pnl = (entry_price - exit_price) / entry_price * notional

                total_fees = entry_commission + exit_commission
                net_pnl = gross_pnl - total_fees

                # update equity
                equity = equity + net_pnl

                # fill trade record
                trades[-1].update({
                    'exit_index': idx,
                    'exit_timestamp': ts,
                    'exit_price': exit_price,
                    'exit_commission': exit_commission,
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'return_pct': net_pnl / entry_equity if entry_equity != 0 else np.nan,
                    'duration': (pd.to_datetime(ts) - pd.to_datetime(trades[-1]['entry_timestamp'])).total_seconds()
                })

                # close position
                in_trade = False
                entry_idx = None
                entry_price = None
                entry_signal = 0
                entry_commission = 0.0
                notional = 0.0

                # If sig != 0 and opposite, we will open new trade on next iteration (or same row if desired)
                # For simplicity we open new one on next loop iteration, consistent with using next bar open.
        
        equity_hist.append(equity)
        equity_index.append(ts)

    # If still in trade at the end, close at last price (df.iloc[-1])
    if in_trade and trades:
        last_row = df.iloc[-1]
        price = last_row['close']
        ts = last_row['timestamp']
        if entry_signal == 1:
            exit_price = price * (1 - (spread_pct / 2.0 + slippage_pct))
        else:
            exit_price = price * (1 + (spread_pct / 2.0 + slippage_pct))
        exit_commission = commission_pct * notional
        if entry_signal == 1:
            gross_pnl = (exit_price - entry_price) / entry_price * notional
        else:
            gross_pnl = (entry_price - exit_price) / entry_price * notional
        total_fees = commission_pct * notional + exit_commission
        net_pnl = gross_pnl - total_fees
        equity = equity + net_pnl
        trades[-1].update({
            'exit_index': df.index[-1],
            'exit_timestamp': ts,
            'exit_price': exit_price,
            'exit_commission': exit_commission,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'return_pct': net_pnl / entry_equity if entry_equity != 0 else np.nan,
            'duration': (pd.to_datetime(ts) - pd.to_datetime(trades[-1]['entry_timestamp'])).total_seconds()
        })
        equity_hist[-1] = equity

    # Build equity series
    equity_series = pd.Series(index=pd.to_datetime(equity_index), data=equity_hist)
    trades_df = pd.DataFrame(trades)

    return trades_df, equity_series

# -------------------------
# Report generation
# -------------------------
def generate_report(trades_df: pd.DataFrame, equity_series: pd.Series, df: pd.DataFrame, config: dict):
    timeframe = config.get('timeframe', '1h')
    periods_per_year = periods_per_year_from_timeframe(timeframe)
    start_capital = config['execution'].get('start_capital', 10000.0)
    rf = config['execution'].get('risk_free_rate', 0.0)

    # Build per-period strategy returns from equity_series
    eq = equity_series.fillna(method='ffill')
    # per-period returns (simple)
    per_period_ret = eq.pct_change().fillna(0)

    total_return = eq.iloc[-1] / eq.iloc[0] - 1.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / (len(eq) / periods_per_year)) - 1 if len(eq) > 0 else np.nan
    ann_vol = per_period_ret.std() * math.sqrt(periods_per_year)
    ann_sharpe = sharpe_ratio(per_period_ret, periods_per_year, rf)
    ann_sortino = sortino_ratio(per_period_ret, periods_per_year, rf)
    md, md_start, md_end = max_drawdown(eq)

    # Trade-level metrics
    trades = trades_df.copy()
    trades['win'] = trades['net_pnl'] > 0
    n_trades = len(trades)
    wins = trades['win'].sum() if n_trades > 0 else 0
    losses = n_trades - wins
    win_rate = wins / n_trades if n_trades > 0 else np.nan
    avg_return_per_trade = trades['return_pct'].mean() if n_trades > 0 else np.nan
    avg_net_pnl = trades['net_pnl'].mean() if n_trades > 0 else np.nan
    avg_gross_pnl = trades['gross_pnl'].mean() if n_trades > 0 else np.nan
    avg_win = trades.loc[trades['win'], 'net_pnl'].mean() if wins > 0 else np.nan
    avg_loss = trades.loc[~trades['win'], 'net_pnl'].mean() if losses > 0 else np.nan
    largest_win = trades['net_pnl'].max() if n_trades > 0 else np.nan
    largest_loss = trades['net_pnl'].min() if n_trades > 0 else np.nan
    percent_profitable = win_rate * 100 if not math.isnan(win_rate) else np.nan

    gross_profit = trades.loc[trades['net_pnl'] > 0, 'net_pnl'].sum() if n_trades > 0 else 0.0
    gross_loss = -trades.loc[trades['net_pnl'] < 0, 'net_pnl'].sum() if n_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan

    # Expectancy = average (win%) * avg_win + (loss%) * avg_loss (in net_pnl terms) normalized per trade
    if n_trades > 0:
        p = win_rate
        avg_win_trade = trades.loc[trades['win'], 'net_pnl'].mean() if wins > 0 else 0.0
        avg_loss_trade = trades.loc[~trades['win'], 'net_pnl'].mean() if losses > 0 else 0.0
        expectancy = p * avg_win_trade + (1 - p) * avg_loss_trade
    else:
        expectancy = np.nan

    # Average trade duration (seconds -> convert to hours)
    avg_duration_sec = trades['duration'].mean() if 'duration' in trades.columns and n_trades > 0 else np.nan
    avg_duration_hours = avg_duration_sec / 3600.0 if not pd.isna(avg_duration_sec) else np.nan

    # Turnover: sum(notional)/average equity over period (approx)
    avg_equity = eq.mean()
    total_turnover = trades['notional'].sum() if 'notional' in trades.columns else 0.0
    turnover_ratio = total_turnover / avg_equity if avg_equity != 0 else np.nan

    # Compile report
    report = {
        'start_time': str(eq.index[0]) if len(eq) else None,
        'end_time': str(eq.index[-1]) if len(eq) else None,
        'periods': len(eq),
        'start_capital': start_capital,
        'ending_capital': float(eq.iloc[-1]) if len(eq) else None,
        'total_return_pct': float(total_return * 100),
        'cagr_pct': float(cagr * 100) if not pd.isna(cagr) else None,
        'annual_vol_pct': float(ann_vol * 100) if not pd.isna(ann_vol) else None,
        'sharpe_ratio': float(ann_sharpe) if not pd.isna(ann_sharpe) else None,
        'sortino_ratio': float(ann_sortino) if not pd.isna(ann_sortino) else None,
        'max_drawdown_pct': float(md * 100) if not pd.isna(md) else None,
        'max_drawdown_start': str(md_start) if md_start is not None else None,
        'max_drawdown_end': str(md_end) if md_end is not None else None,
        'total_trades': int(n_trades),
        'wins': int(wins),
        'losses': int(losses),
        'win_rate_pct': float(percent_profitable) if not pd.isna(percent_profitable) else None,
        'avg_return_per_trade_pct': float(avg_return_per_trade * 100) if not pd.isna(avg_return_per_trade) else None,
        'avg_net_pnl': float(avg_net_pnl) if not pd.isna(avg_net_pnl) else None,
        'avg_gross_pnl': float(avg_gross_pnl) if not pd.isna(avg_gross_pnl) else None,
        'avg_win': float(avg_win) if not pd.isna(avg_win) else None,
        'avg_loss': float(avg_loss) if not pd.isna(avg_loss) else None,
        'largest_win': float(largest_win) if not pd.isna(largest_win) else None,
        'largest_loss': float(largest_loss) if not pd.isna(largest_loss) else None,
        'profit_factor': float(profit_factor) if not pd.isna(profit_factor) else None,
        'expectancy': float(expectancy) if not pd.isna(expectancy) else None,
        'avg_trade_duration_hours': float(avg_duration_hours) if not pd.isna(avg_duration_hours) else None,
        'turnover_ratio': float(turnover_ratio) if not pd.isna(turnover_ratio) else None
    }

    return report

# -------------------------
# Main flow
# -------------------------
def main(config_path='config.json', data_csv_path=None, lookback=20):
    # Load config
    config = {}
    with open(config_path, 'r') as f:
        config = json.load(f)

    # If data CSV explicitly passed, override JSON output filename
    if data_csv_path:
        csv_path = Path(data_csv_path)
    else:
        csv_path = Path(config['output'].get('filename', 'data.csv'))

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}. Provide a CSV or fetch data first.")

    # Read CSV
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    # Ensure numeric
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Apply Donchian breakout (use lookback from config if provided)
    lookback = config.get('strategy', {}).get('lookback', lookback)
    df_signal = donchian_breakout(df, lookback)

    # Simulate trades
    trades_df, equity_series = simulate_trades(df_signal, config)

    # Generate report
    report = generate_report(trades_df, equity_series, df_signal, config)

    # Save outputs
    out_dir = Path(config.get('output', {}).get('dir', '.'))
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_file = out_dir / config.get('output', {}).get('trades_filename', 'trade_log.csv')
    metrics_file = out_dir / config.get('output', {}).get('metrics_filename', 'metrics.json')
    equity_plot_file = out_dir / config.get('output', {}).get('equity_plot', 'equity_curve.png')

    # Save trade log (expand timestamps)
    if len(trades_df) > 0:
        trades_df.to_csv(trades_file, index=False)
    else:
        pd.DataFrame().to_csv(trades_file, index=False)

    with open(metrics_file, 'w') as f:
        json.dump({'report': report}, f, indent=2, default=str)

    # Plot equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(equity_series.index, equity_series.values)
    plt.title('Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(equity_plot_file)
    plt.close()

    # Print summary
    print("==== Backtest Summary ====")
    print(f"Start: {report['start_time']}  End: {report['end_time']}")
    print(f"Start capital: {report['start_capital']:.2f}   End capital: {report['ending_capital']:.2f}")
    print(f"Total return: {report['total_return_pct']:.2f}%   CAGR: {report['cagr_pct']:.2f}%")
    print(f"Sharpe: {report['sharpe_ratio']:.2f}   MaxDD: {report['max_drawdown_pct']:.2f}%")
    print(f"Trades: {report['total_trades']}  Win rate: {report['win_rate_pct']:.2f}%  Profit factor: {report['profit_factor']}")
    print(f"Trade log saved to: {trades_file}")
    print(f"Metrics JSON saved to: {metrics_file}")
    print(f"Equity plot saved to: {equity_plot_file}")

    return trades_df, report, equity_series

if __name__ == "__main__":
    # default config path
    trades, report, eq = main(config_path='config.json')
