# Caching System Guide

## Overview

The vibestock package now includes an intelligent caching system that dramatically improves performance for repeated analyses by storing downloaded stock data locally.

## How It Works

### Automatic Caching

When you run `python main.py`, the system:

1. **First Run**: Downloads all S&P 500 data and saves it to `cache/price_cache.pkl`
2. **Subsequent Runs**: Checks if cached data exists and is up-to-date
   - If valid: Uses cached data (10-30 seconds)
   - If invalid: Downloads fresh data (2-5 minutes)

### Cache Validation Rules

Cache is considered **valid** when:
- Cache file exists
- Cache was downloaded within the last 24 hours
- Cached data is within 3 days of the requested end date

Cache is considered **invalid** when:
- No cache file exists
- Cache is older than 24 hours
- Cached data is more than 3 days behind the requested date
- Requested start date is earlier than cached start date

## Usage Examples

### Basic Usage (with automatic caching)

```bash
# First run - downloads and caches data
python main.py
# Output: Downloading fresh stock data... (2-5 minutes)

# Second run - uses cache
python main.py
# Output: Using cached price data (10-30 seconds)
```

### Force Re-download

```bash
# Bypass cache and download fresh data
python main.py --force-download
```

### Clear Cache

```bash
# Remove cached data
python main.py --clear-cache
```

### Different Time Periods

```bash
# Analyze 5 years of data
python main.py --days 1825

# If cache has 2 years but you request 5 years, it will re-download
```

## Cache File Details

### Location
- **Path**: `cache/price_cache.pkl`
- **Format**: Python pickle file

### Contents
The cache file stores:
```python
{
    'prices': DataFrame,           # Price data for all stocks
    'tickers': list,              # List of ticker symbols
    'start_date': str,            # Start date (YYYY-MM-DD)
    'end_date': str,              # End date (YYYY-MM-DD)
    'download_time': str,         # When cache was created (ISO format)
    'last_data_date': str         # Last date in the dataset
}
```

### Size
- Typical size: 5-15 MB for 2 years of S&P 500 data
- Grows with more stocks and longer time periods

## Performance Comparison

| Operation | Without Cache | With Cache | Speedup |
|-----------|--------------|------------|---------|
| Full S&P 500 analysis (2 years) | 3-5 minutes | 10-30 seconds | **10-15x faster** |
| Individual stock analysis (10 stocks) | 30-60 seconds | 5-10 seconds | **6x faster** |

## Programmatic Usage

You can use the caching functions in your own scripts:

```python
from vibestock import get_or_download_prices, clear_cache

# Get prices with automatic caching
prices = get_or_download_prices(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2023-01-01',
    end_date='2024-10-29',
    force_download=False  # Use cache if valid
)

# Clear cache when needed
clear_cache()
```

### Manual Cache Management

```python
from vibestock.utils import (
    load_price_cache,
    save_price_cache,
    is_cache_valid,
    clear_cache
)

# Load cache
cache_data = load_price_cache('cache')

# Check if valid
if is_cache_valid(cache_data, '2024-10-29', max_age_days=1):
    prices = cache_data['prices']
    print(f"Using cached data: {len(prices)} rows")

# Save cache manually
save_price_cache(prices, tickers, start_date, end_date, 'cache')

# Clear cache
clear_cache('cache')
```

## Best Practices

### When to Use Cache
✅ Daily analysis of the same time period
✅ Testing and development
✅ Repeated analyses with same parameters
✅ When network is slow or unreliable

### When to Force Download
❌ S&P 500 composition changes (new stocks added/removed)
❌ Need to verify data accuracy
❌ Suspect data corruption
❌ After market close to get latest prices

## Troubleshooting

### Cache Won't Load
**Problem**: Error loading cache file

**Solutions**:
```bash
# Clear and rebuild cache
python main.py --clear-cache
python main.py --force-download
```

### Cache Always Invalid
**Problem**: Cache keeps re-downloading even though it's recent

**Cause**: You may be requesting a longer time period than cached

**Solution**: Ensure consistent time periods or clear and rebuild:
```bash
python main.py --clear-cache
python main.py --days 730  # Use consistent time period
```

### Disk Space Issues
**Problem**: Cache files taking up too much space

**Solution**: Clear cache periodically:
```bash
# Manual cleanup
python main.py --clear-cache

# Or delete cache directory
rm -rf cache/
```

## Configuration

You can customize cache behavior by modifying `vibestock/utils.py`:

```python
# Change cache validity period (default: 1 day)
is_cache_valid(cache_data, end_date, max_age_days=7)  # 7 days

# Change cache directory
get_or_download_prices(tickers, start, end, cache_dir='my_cache')
```

## Security & Privacy

- Cache files are stored **locally only**
- No data is sent to external servers (except Yahoo Finance for downloads)
- Cache contains only public market data
- Safe to commit to version control (but excluded via `.gitignore`)

## Maintenance

### Regular Cleanup
```bash
# Clear cache once a week for fresh data
python main.py --clear-cache
python main.py
```

### Automated Workflow
```bash
#!/bin/bash
# Daily analysis script with weekly cache refresh

# Get day of week (0-6, 0 is Sunday)
day=$(date +%w)

# Clear cache on Sundays
if [ $day -eq 0 ]; then
    python main.py --clear-cache
fi

# Run analysis
python main.py
```

## FAQ

**Q: Does cache work offline?**
A: Yes, if cache is valid, you don't need internet connection for analysis.

**Q: Can I share cache between projects?**
A: Yes, use custom cache directories:
```python
prices = get_or_download_prices(tickers, start, end, cache_dir='/shared/cache')
```

**Q: What happens if stock is delisted?**
A: Old cache may contain delisted stocks. Force re-download to get current S&P 500.

**Q: Is cache thread-safe?**
A: No, avoid running multiple analyses simultaneously. Use separate cache directories.

## Advanced: Cache Inspection

```python
import pickle
import pandas as pd

# Load and inspect cache
with open('cache/price_cache.pkl', 'rb') as f:
    cache = pickle.load(f)

print(f"Tickers: {len(cache['tickers'])}")
print(f"Date range: {cache['start_date']} to {cache['end_date']}")
print(f"Downloaded: {cache['download_time']}")
print(f"Shape: {cache['prices'].shape}")
print(cache['prices'].head())
```

