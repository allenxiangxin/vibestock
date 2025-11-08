# S&P 500 Return Distribution Analyzer

A comprehensive tool for analyzing return distributions across S&P 500 stocks using professional-grade Polygon.io data.

## Overview

This tool automatically fetches the current S&P 500 ticker list and performs sophisticated statistical analysis on historical stock returns, complete with visualization and caching capabilities.

## Features

- **Professional Data Source**: Uses Polygon.io API for reliable, high-quality stock data
- **Smart Caching System**: Caches downloaded data locally for instant subsequent analyses
- **Comprehensive Statistics**: Mean, median, std dev, skewness, kurtosis, normality testing
- **Beautiful Visualizations**: Distribution plots, Q-Q plots, correlation heatmaps, and more
- **Flexible Analysis**: Supports daily, weekly, or monthly return calculations
- **Modular Architecture**: Clean separation of concerns for easy customization

## Quick Start

### Prerequisites

1. Python 3.8+ with conda environment
2. Polygon.io API key (free tier available at https://polygon.io/)
3. API key configured in `.env` file at project root

### Installation

```bash
# Activate environment (from project root)
conda activate vibestock

# Install dependencies
pip install -r ../requirements.txt
```

### Basic Usage

```bash
# Run full S&P 500 analysis (from this directory)
python main.py

# Or from project root
cd /path/to/vibestock
python sp500_analyzer/main.py
```

### Command-Line Options

```bash
# Force re-download even if cache is valid
python main.py --force-download

# Analyze 5 years of data instead of 2
python main.py --days 1825

# Use weekly returns
python main.py --return-type weekly

# Combine options
python main.py --days 1095 --return-type monthly
```

## Project Structure

```
sp500_analyzer/
├── README.md               # This file
├── main.py                 # Main entry point
├── vibestock/              # Core package
│   ├── __init__.py        # Package initialization
│   ├── data_fetcher.py    # Data download and ticker fetching
│   ├── analyzer.py        # Statistical analysis
│   ├── visualizer.py      # Visualization generation
│   └── utils.py           # Caching and utilities
└── docs/                   # Documentation
    ├── CACHING_GUIDE.md   # Detailed caching documentation
    ├── FREE_TIER_GUIDE.md # Free tier optimization tips
    └── RATE_LIMIT_FIX.md  # Troubleshooting rate limits
```

## Output

All results are saved to `output/` directory (auto-created):

### Data Files
- `summary_statistics.csv` - Per-stock statistics
- `returns_data.csv` - Full returns dataset
- `top_performers.csv` - Top 50 stocks by return

### Visualizations
- `return_distribution_analysis.png` - Multi-panel distribution analysis
- `cumulative_returns.png` - Portfolio performance over time
- `correlation_matrix.png` - Stock correlation heatmap

## Performance

- **First run (free tier)**: ~100 minutes (one-time download of ~500 stocks)
  - Resume capability: Stop anytime with Ctrl+C, resume later
  - Progress tracking with ETA and countdown
- **Subsequent runs**: 10-30 seconds using cached data ⚡
- **First run (paid tier)**: 2-5 minutes

## Free Tier Optimization

The tool is fully optimized for Polygon.io's free tier (5 API calls/minute):

1. **Smart Caching**: Downloads once, uses cache forever
2. **Resume Support**: Stop and resume anytime without losing progress
3. **Incremental Updates**: Only downloads new/missing stocks
4. **Progress Tracking**: Real-time ETA and countdown timers

See [docs/FREE_TIER_GUIDE.md](docs/FREE_TIER_GUIDE.md) for more details.

## Configuration

### API Key Setup

Create a `.env` file in the **project root** (not this directory):

```bash
POLYGON_API_KEY=your_actual_api_key_here
```

See `../../docs/POLYGON_SETUP.md` for detailed instructions.

### Adjust Rate Limits

Default settings (free tier - 5 calls/min):
```bash
python main.py  # Uses 12 second delay
```

For paid tiers:
```bash
# Starter (100 calls/min)
python main.py --delay 0.6

# Developer (unlimited)
python main.py --delay 0.1
```

## Using as a Library

```python
from vibestock import (
    get_sp500_tickers,
    get_or_download_prices,
    calculate_returns,
    analyze_return_distribution,
    create_visualizations
)

# Get tickers
tickers = get_sp500_tickers()

# Download/load data with smart caching
prices = get_or_download_prices(
    tickers, 
    start_date='2022-01-01',
    end_date='2024-01-01'
)

# Analyze
returns = calculate_returns(prices, return_type='daily')
stats = analyze_return_distribution(returns)

# Visualize
create_visualizations(returns, stats, output_dir='output')
```

## Documentation

- **[CACHING_GUIDE.md](docs/CACHING_GUIDE.md)** - How caching works
- **[FREE_TIER_GUIDE.md](docs/FREE_TIER_GUIDE.md)** - Optimize for free tier
- **[RATE_LIMIT_FIX.md](docs/RATE_LIMIT_FIX.md)** - Troubleshooting guide

## Troubleshooting

### "API key not found"
- Ensure `.env` file exists in project root (not this directory)
- Verify it contains: `POLYGON_API_KEY=your_key`
- No quotes or spaces

### Slow downloads
- Normal for free tier (5 calls/min = ~100 min first run)
- Use Ctrl+C to stop, resume anytime
- Upgrade to paid tier for faster downloads

### Memory issues
- Reduce date range: `python main.py --days 365`
- Use weekly/monthly returns: `python main.py --return-type weekly`

## Related Tools

- **[../gold_tracker/](../gold_tracker/)** - Gold price MACD tracker
- **[../docs/](../docs/)** - Shared documentation

---

**Part of the vibestock project** - See [main README](../README.md) for full project documentation.

