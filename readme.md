# Cryptocurrency Trading System with Donchian Channel Strategy

This project consists of two main components that work together to create a complete trading system:
1. **Data Fetcher** - Downloads historical cryptocurrency data from Binance
2. **Backtester** - Tests a Donchian Channel breakout strategy on the downloaded data

## 📁 Project Structure

```
├── fetch_data.py           # Downloads crypto data from Binance
├── donchian_backtest_report.py  # Runs strategy backtests
├── main.py             # Runs both scripts in sequence
├── config.json            # Configuration file (you need to create this)
└── README.md              # This file
```

## 🚀 Quick Start

### Option 1: Run Everything at Once
```bash
python main.py
```
This will:
1. Download the data based on your config
2. Run the backtest automatically
3. Generate all reports and charts

### Option 2: Run Individual Components

**Step 1: Download Data**
```bash
python Fetch_data.py
```

**Step 2: Run Backtest**
```bash
python backtest.py
```

## ⚙️ Configuration File (config.json)

You need to create a `config.json` file with your settings. Here's what each parameter means:

### Data Fetching Settings

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "limit": 1000,
  "fetch_all": false,
  "max_requests": 100,
  "start_time": 1609459200000,
  "end_time": null,
  
  "api": {
    "base_url": "https://api.binance.com",
    "endpoint": "/api/v3/klines"
  },
  
  "output": {
    "filename": "btc_data.csv",
    "format": "csv",
    "dir": "./results",
    "trades_filename": "trade_log.csv",
    "metrics_filename": "metrics.json",
    "equity_plot": "equity_curve.png"
  },
  
  "strategy": {
    "lookback": 20
  },
  
  "execution": {
    "start_capital": 10000.0,
    "position_size_fraction": 1.0,
    "leverage": 1.0,
    "commission_pct": 0.001,
    "slippage_pct": 0.0005,
    "spread_pct": 0.0002,
    "risk_free_rate": 0.0
  }
}
```

## 📊 Parameter Explanations

### Data Parameters

| Parameter | What it does | Example | Why it matters |
|-----------|--------------|---------|----------------|
| `symbol` | Which cryptocurrency to download | "BTCUSDT" | Different coins have different volatility patterns |
| `timeframe` | Time interval for each data point | "1h", "15m", "1d" | Shorter timeframes = more trades, longer = less noise |
| `limit` | How many data points to fetch | 1000 | More data = longer backtests but slower processing |
| `fetch_all` | Download all available history | true/false | true = gets years of data, false = gets recent data only |
| `max_requests` | Limit API calls to avoid rate limits | 100 | Higher = more data but risk of getting blocked |

### Strategy Parameters

| Parameter | What it does | Typical Range | Impact |
|-----------|--------------|---------------|---------|
| `lookback` | Donchian channel period | 10-50 | Smaller = more sensitive, more trades; Larger = less noise, fewer trades |

**Donchian Channel Explained:**
- Looks at the highest high and lowest low over the last X periods
- **Buy Signal**: Price breaks above the highest high
- **Sell Signal**: Price breaks below the lowest low
- Think of it as trading breakouts from recent trading ranges

### Execution Parameters

| Parameter | What it controls | Typical Value | Real-world meaning |
|-----------|------------------|---------------|-------------------|
| `start_capital` | Starting money | 10000.0 | Your initial investment amount |
| `position_size_fraction` | How much of your money to risk per trade | 0.5-1.0 | 1.0 = use all available capital, 0.5 = use half |
| `leverage` | Borrowing multiplier | 1.0-10.0 | 2.0 = double your position size (more risk/reward) |
| `commission_pct` | Trading fees | 0.001 (0.1%) | What your broker charges per trade |
| `slippage_pct` | Market impact cost | 0.0005 (0.05%) | Price moves against you when placing large orders |
| `spread_pct` | Bid-ask spread cost | 0.0002 (0.02%) | Difference between buying and selling price |

## 📈 Understanding the Results

The backtest generates several files:

### 1. Trade Log (trade_log.csv)
Shows every individual trade with:
- Entry/exit times and prices
- Profit/loss for each trade
- Position sizes and commissions

### 2. Performance Metrics (metrics.json)
Key metrics to understand:

| Metric | What it means | Good value |
|--------|---------------|------------|
| `total_return_pct` | Total profit/loss percentage | Positive numbers |
| `cagr_pct` | Compound annual growth rate | > 10% is good |
| `sharpe_ratio` | Risk-adjusted returns | > 1.0 is good, > 2.0 is excellent |
| `max_drawdown_pct` | Worst losing streak | < 20% is manageable |
| `win_rate_pct` | Percentage of winning trades | > 50% is good but not required |
| `profit_factor` | Total wins ÷ Total losses | > 1.25 is good |

### 3. Equity Curve (equity_curve.png)
Visual chart showing how your account balance changed over time. Look for:
- **Upward trend**: Strategy is profitable
- **Smooth curve**: Consistent performance
- **Sharp drops**: Major losing periods (drawdowns)

## 🛠️ Installation Requirements

```bash
pip install pandas numpy matplotlib scipy requests
```

## ⚡ Performance Tips

### For Faster Data Download:
- Set `fetch_all: false` for recent data only
- Use larger timeframes ("1d" instead of "1h")
- Reduce `max_requests` if you don't need years of data

### For Faster Backtesting:
- Use smaller datasets
- Test fewer parameter combinations
- Start with shorter time periods

## 🎯 Strategy Optimization

### Lookback Period Effects:
- **Small values (5-15)**: More trades, higher turnover, catches short-term moves
- **Large values (30-50)**: Fewer trades, follows major trends, less noise

### Position Sizing Impact:
- **Conservative (0.5)**: Lower risk, smoother equity curve
- **Aggressive (1.0)**: Higher potential returns, more volatility

