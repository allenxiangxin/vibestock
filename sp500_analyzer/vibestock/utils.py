"""
Utilities Module

Helper functions for saving results and other utilities.
"""

import os
import pandas as pd
import pickle
from datetime import datetime, timedelta


def save_results(summary_stats, returns, output_dir='output'):
    """
    Save analysis results to CSV files.
    
    Args:
        summary_stats (pd.DataFrame): Summary statistics DataFrame
        returns (pd.DataFrame): Returns DataFrame
        output_dir (str): Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("SAVING RESULTS...")
    print("="*80)
    
    # Save summary statistics
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_stats.to_csv(summary_path)
    print(f"✓ Saved summary statistics: {summary_path}")
    
    # Save full returns data
    returns_path = os.path.join(output_dir, 'returns_data.csv')
    returns.to_csv(returns_path)
    print(f"✓ Saved returns data: {returns_path}")
    
    # Save top performers
    top_path = os.path.join(output_dir, 'top_performers.csv')
    top_df = summary_stats.T.sort_values('Mean', ascending=False).head(50)
    top_df.to_csv(top_path)
    print(f"✓ Saved top 50 performers: {top_path}")


def create_summary_report(summary_stats, returns, output_dir='output'):
    """
    Create a text summary report.
    
    Args:
        summary_stats (pd.DataFrame): Summary statistics DataFrame
        returns (pd.DataFrame): Returns DataFrame
        output_dir (str): Directory to save report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("S&P 500 STOCK RETURN DISTRIBUTION ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("ANALYSIS OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Total stocks analyzed: {len(summary_stats.columns)}\n")
        f.write(f"Date range: {returns.index[0]} to {returns.index[-1]}\n")
        f.write(f"Total trading days: {len(returns)}\n\n")
        
        # Top performers
        f.write("TOP 10 PERFORMERS (by Mean Return)\n")
        f.write("-"*80 + "\n")
        top = summary_stats.T.sort_values('Mean', ascending=False).head(10)
        for i, (ticker, row) in enumerate(top.iterrows(), 1):
            f.write(f"{i:2d}. {ticker:6s}: Mean={row['Mean']*100:.4f}%, Std={row['Std Dev']*100:.4f}%\n")
        
        f.write("\n")
        
        # Bottom performers
        f.write("BOTTOM 10 PERFORMERS (by Mean Return)\n")
        f.write("-"*80 + "\n")
        bottom = summary_stats.T.sort_values('Mean', ascending=True).head(10)
        for i, (ticker, row) in enumerate(bottom.iterrows(), 1):
            f.write(f"{i:2d}. {ticker:6s}: Mean={row['Mean']*100:.4f}%, Std={row['Std Dev']*100:.4f}%\n")
    
    print(f"✓ Saved summary report: {report_path}")


def load_cached_data(output_dir='output'):
    """
    Load previously saved data if available.
    
    Args:
        output_dir (str): Directory where data is saved
        
    Returns:
        tuple: (returns, summary_stats) or (None, None) if not found
    """
    returns_path = os.path.join(output_dir, 'returns_data.csv')
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    
    if os.path.exists(returns_path) and os.path.exists(summary_path):
        print("Loading cached data...")
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        summary_stats = pd.read_csv(summary_path, index_col=0)
        print(f"✓ Loaded cached data for {len(returns.columns)} stocks")
        return returns, summary_stats
    
    return None, None


def save_price_cache(prices, tickers, start_date, end_date, cache_dir='cache'):
    """
    Save downloaded price data to cache with metadata.
    
    Args:
        prices (pd.DataFrame): Price data
        tickers (list): List of tickers
        start_date (str): Start date
        end_date (str): End date
        cache_dir (str): Directory to save cache
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_data = {
        'prices': prices,
        'tickers': tickers,
        'start_date': start_date,
        'end_date': end_date,
        'download_time': datetime.now().isoformat(),
        'last_data_date': prices.index[-1].isoformat() if not prices.empty else None
    }
    
    cache_path = os.path.join(cache_dir, 'price_cache.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"✓ Saved price data to cache: {cache_path}")
    print(f"  - {len(prices.columns)} stocks, {len(prices)} days")
    print(f"  - Date range: {prices.index[0].date()} to {prices.index[-1].date()}")


def load_price_cache(cache_dir='cache'):
    """
    Load cached price data if available.
    
    Args:
        cache_dir (str): Directory where cache is stored
        
    Returns:
        dict: Cache data or None if not found
    """
    cache_path = os.path.join(cache_dir, 'price_cache.pkl')
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        return cache_data
    except Exception as e:
        print(f"Warning: Failed to load cache: {e}")
        return None


def is_cache_valid(cache_data, required_end_date, max_age_days=1):
    """
    Check if cached data is up to date.
    
    Args:
        cache_data (dict): Cached data
        required_end_date (str): Required end date (YYYY-MM-DD)
        max_age_days (int): Maximum age of cache in days
        
    Returns:
        bool: True if cache is valid, False otherwise
    """
    if cache_data is None:
        return False
    
    try:
        # Check if cache has data
        if cache_data['prices'].empty:
            print("Cache is empty")
            return False
        
        # Parse dates
        last_data_date = pd.to_datetime(cache_data['last_data_date'])
        required_date = pd.to_datetime(required_end_date)
        download_time = pd.to_datetime(cache_data['download_time'])
        
        # Check cache age
        cache_age = datetime.now() - download_time
        if cache_age.days > max_age_days:
            print(f"Cache is too old ({cache_age.days} days)")
            return False
        
        # Check if data is up to date
        # Allow 1 day tolerance for weekends/holidays
        date_diff = (required_date - last_data_date).days
        if date_diff > 3:  # More than 3 days behind
            print(f"Cache data is outdated (last date: {last_data_date.date()}, need: {required_date.date()})")
            return False
        
        print(f"✓ Cache is valid (last updated: {download_time.strftime('%Y-%m-%d %H:%M')})")
        print(f"  - Last data date: {last_data_date.date()}")
        print(f"  - {len(cache_data['prices'].columns)} stocks cached")
        return True
        
    except Exception as e:
        print(f"Error validating cache: {e}")
        return False


def get_or_download_prices(tickers, start_date, end_date, force_download=False, 
                          cache_dir='cache', batch_size=100, delay=12.0):
    """
    Get prices from cache or download if needed.
    Implements smart incremental downloading - only downloads missing stocks.
    
    Args:
        tickers (list): List of tickers
        start_date (str): Start date
        end_date (str): End date
        force_download (bool): Force re-download even if cache is valid
        cache_dir (str): Cache directory
        batch_size (int): Number of stocks per batch for downloads
        delay (float): Delay between API calls in seconds (12.0 for free tier = 5 calls/min)
        
    Returns:
        pd.DataFrame: Price data
    """
    from .data_fetcher import download_stock_data
    
    # Try to load from cache
    if not force_download:
        cache_data = load_price_cache(cache_dir)
        if cache_data and is_cache_valid(cache_data, end_date):
            print("✓ Found cached price data")
            
            # Check if we need more historical data
            cached_start = pd.to_datetime(cache_data['start_date'])
            requested_start = pd.to_datetime(start_date)
            
            if requested_start < cached_start:
                print(f"⚠ Requested start date {start_date} is earlier than cached {cache_data['start_date']}")
                print("Need to re-download with extended date range")
            else:
                # Check which tickers are missing from cache
                cached_tickers = set(cache_data['prices'].columns)
                requested_tickers = set(tickers)
                missing_tickers = requested_tickers - cached_tickers
                
                if missing_tickers:
                    print(f"\n📥 Found {len(missing_tickers)} new stocks not in cache")
                    print(f"   Cached: {len(cached_tickers)} stocks")
                    print(f"   Downloading: {len(missing_tickers)} new stocks")
                    
                    # Download only missing stocks
                    new_prices = download_stock_data(
                        list(missing_tickers), 
                        start_date, 
                        end_date,
                        batch_size=batch_size,
                        delay=delay
                    )
                    
                    if not new_prices.empty:
                        # Merge with cached data
                        print(f"\n✓ Merging {len(new_prices.columns)} new stocks with cache...")
                        combined_prices = pd.concat([cache_data['prices'], new_prices], axis=1)
                        
                        # Save updated cache
                        all_tickers = list(cached_tickers) + list(new_prices.columns)
                        save_price_cache(combined_prices, all_tickers, start_date, end_date, cache_dir)
                        
                        # Filter to requested tickers and date range
                        prices = combined_prices[combined_prices.index >= start_date]
                        return prices
                
                # All tickers are in cache
                print(f"✓ All {len(requested_tickers)} stocks found in cache")
                prices = cache_data['prices']
                
                # Filter to only requested tickers
                available_tickers = [t for t in tickers if t in prices.columns]
                if len(available_tickers) < len(tickers):
                    missing = len(tickers) - len(available_tickers)
                    print(f"⚠ {missing} requested stocks not available in cache")
                
                prices = prices[available_tickers]
                prices = prices[prices.index >= start_date]
                return prices
    
    # Download fresh data
    print("\n📥 Downloading fresh stock data...")
    print(f"⏱️  Free tier: 5 calls/min = {delay} seconds between calls")
    print(f"⏱️  Estimated time: ~{len(tickers) * delay / 60:.1f} minutes for {len(tickers)} stocks")
    print("💡 Tip: Once downloaded, data is cached - future runs will be instant!")
    
    prices = download_stock_data(tickers, start_date, end_date, 
                                 batch_size=batch_size, delay=delay)
    
    # Save to cache
    if not prices.empty:
        save_price_cache(prices, tickers, start_date, end_date, cache_dir)
    
    return prices


def clear_cache(cache_dir='cache'):
    """
    Clear all cached data.
    
    Args:
        cache_dir (str): Cache directory
    """
    cache_path = os.path.join(cache_dir, 'price_cache.pkl')
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"✓ Cache cleared: {cache_path}")
    else:
        print("No cache to clear")

