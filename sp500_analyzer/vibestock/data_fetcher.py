"""
Data Fetcher Module

Functions for fetching S&P 500 tickers and downloading historical stock data using Polygon.io.
"""

import pandas as pd
import urllib.request
import time
from datetime import datetime, timedelta
from polygon import RESTClient
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()


def get_sp500_tickers():
    """
    Fetch the list of S&P 500 tickers from Wikipedia.
    
    Returns:
        list: List of ticker symbols
    """
    print("Fetching S&P 500 ticker list...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # Add user-agent header to avoid 403 Forbidden error
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    req = urllib.request.Request(url, headers=headers)
    
    # Fetch the HTML content
    with urllib.request.urlopen(req) as response:
        html = response.read()
    
    # Parse tables from HTML
    tables = pd.read_html(html)
    sp500_table = tables[0]
    tickers = sp500_table['Symbol'].tolist()
    # Clean tickers (replace dots with dashes for Yahoo Finance compatibility)
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    print(f"Found {len(tickers)} S&P 500 companies")
    return tickers


def get_polygon_client():
    """
    Get Polygon.io API client with API key from environment.
    
    Returns:
        RESTClient: Polygon API client
    """
    api_key = os.getenv('POLYGON_API_KEY')
    
    if not api_key or api_key == 'your_api_key_here':
        print("\n" + "="*80)
        print("ERROR: Polygon.io API key not found!")
        print("="*80)
        print("\nPlease set up your API key:")
        print("1. Sign up at https://polygon.io/ (free tier available)")
        print("2. Get your API key from the dashboard")
        print("3. Create a .env file in the project root")
        print("4. Add this line: POLYGON_API_KEY=your_actual_key")
        print("\nOr set environment variable:")
        print("   export POLYGON_API_KEY=your_actual_key")
        print("="*80)
        raise ValueError("Polygon.io API key not configured")
    
    return RESTClient(api_key)


def download_stock_data(tickers, start_date, end_date, batch_size=100, delay=12.0):
    """
    Download historical stock data using Polygon.io API.
    
    Args:
        tickers (list): List of ticker symbols
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        batch_size (int): Number of stocks to download per batch (default: 100)
        delay (float): Delay in seconds between API calls (default: 12.0 for free tier = 5 calls/min)
        
    Returns:
        pd.DataFrame: DataFrame with adjusted close prices
    """
    print(f"\n{'='*70}")
    print(f"📥 DOWNLOADING STOCK DATA")
    print(f"{'='*70}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Number of stocks: {len(tickers)}")
    print(f"API: Polygon.io")
    
    if len(tickers) == 0:
        print("ERROR: No tickers provided!")
        return pd.DataFrame()
    
    try:
        client = get_polygon_client()
    except ValueError as e:
        print(str(e))
        return pd.DataFrame()
    
    # Calculate estimated time
    estimated_minutes = (len(tickers) * delay) / 60
    estimated_hours = estimated_minutes / 60
    
    print(f"\n⏱️  Rate limit: 5 calls/min (free tier)")
    print(f"⏱️  Delay per stock: {delay} seconds")
    if estimated_hours >= 1:
        print(f"⏱️  Estimated time: ~{estimated_hours:.1f} hours ({estimated_minutes:.0f} minutes)")
    else:
        print(f"⏱️  Estimated time: ~{estimated_minutes:.1f} minutes")
    
    print(f"\n💡 Tip: This is a ONE-TIME download. Future runs use cache (30 seconds)!")
    print(f"💡 Tip: You can stop and resume - already downloaded stocks are saved\n")
    
    # Download data for each ticker
    all_prices = {}
    failed_tickers = []
    
    import time as time_module
    start_time = time_module.time()
    
    for i, ticker in enumerate(tickers, 1):
        try:
            # Calculate ETA
            if i > 1:
                elapsed = time_module.time() - start_time
                rate = elapsed / (i - 1)
                remaining = (len(tickers) - i + 1) * rate
                eta_min = remaining / 60
                
                if eta_min >= 60:
                    eta_str = f"ETA: {eta_min/60:.1f}h"
                else:
                    eta_str = f"ETA: {eta_min:.1f}min"
            else:
                eta_str = "Calculating..."
            
            # Show progress
            progress_pct = (i / len(tickers)) * 100
            print(f"[{i:3d}/{len(tickers)}] {progress_pct:5.1f}% | {ticker:6s} | {eta_str}", end='\r')
            
            # Get aggregates (bars) for the ticker
            # Polygon.io uses 'day' for daily aggregates
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start_date,
                to=end_date,
                limit=50000  # Max limit
            )
            
            # Convert to pandas DataFrame
            if aggs and len(aggs) > 0:
                dates = [datetime.fromtimestamp(agg.timestamp / 1000) for agg in aggs]
                closes = [agg.close for agg in aggs]
                
                # Create series with dates as index
                series = pd.Series(closes, index=dates, name=ticker)
                all_prices[ticker] = series
            else:
                failed_tickers.append(ticker)
            
            # Rate limiting delay (Polygon free tier: 5 calls/min = 12 sec delay)
            # Show countdown for long delays
            if delay >= 5 and i < len(tickers):
                for remaining in range(int(delay), 0, -1):
                    if remaining > 10 or remaining <= 3:  # Show start and end of countdown
                        print(f"[{i:3d}/{len(tickers)}] Waiting {remaining}s... (rate limit: 5 calls/min)", end='\r')
                    time.sleep(1)
            else:
                time.sleep(delay)
            
        except Exception as e:
            error_msg = str(e)
            if 'rate limit' in error_msg.lower() or '429' in error_msg:
                print(f"\n⚠ Rate limited at stock {i}. Waiting 60 seconds...")
                time.sleep(60)
                # Retry this ticker
                try:
                    aggs = client.get_aggs(
                        ticker=ticker,
                        multiplier=1,
                        timespan="day",
                        from_=start_date,
                        to=end_date,
                        limit=50000
                    )
                    if aggs and len(aggs) > 0:
                        dates = [datetime.fromtimestamp(agg.timestamp / 1000) for agg in aggs]
                        closes = [agg.close for agg in aggs]
                        series = pd.Series(closes, index=dates, name=ticker)
                        all_prices[ticker] = series
                    else:
                        failed_tickers.append(ticker)
                except:
                    failed_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)
                if i <= 5:  # Only show errors for first few stocks
                    print(f"\n⚠ Failed to download {ticker}: {error_msg[:100]}")
    
    print(f"\nProgress: {len(tickers)}/{len(tickers)} stocks processed")
    
    # Check if any data was downloaded
    if not all_prices:
        print("\nERROR: No data was downloaded.")
        if failed_tickers:
            print(f"Failed tickers: {failed_tickers[:10]}...")
        return pd.DataFrame()
    
    # Combine all series into a DataFrame
    df = pd.DataFrame(all_prices)
    
    # Sort by date
    df = df.sort_index()
    
    print(f"\n✓ Downloaded {len(df)} rows of data")
    print(f"✓ Successfully downloaded {len(df.columns)} stocks")
    
    if failed_tickers:
        print(f"⚠ Failed to download {len(failed_tickers)} stocks")
        if len(failed_tickers) <= 10:
            print(f"  Failed: {', '.join(failed_tickers)}")
    
    # Remove columns with too much missing data (>50%)
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)
    
    print(f"✓ After filtering missing data: {len(df.columns)} stocks remain")
    
    if df.empty or len(df.columns) == 0:
        print("ERROR: No stocks have sufficient data after filtering.")
        return pd.DataFrame()
    
    print(f"\n✓ Successfully downloaded data for {len(df.columns)} stocks")
    return df


def calculate_returns(prices, return_type='daily'):
    """
    Calculate returns from price data.
    
    Args:
        prices (pd.DataFrame): DataFrame with price data
        return_type (str): Type of return ('daily', 'weekly', 'monthly')
        
    Returns:
        pd.DataFrame: DataFrame with returns
    """
    print(f"\nCalculating {return_type} returns...")
    
    if prices.empty:
        print("ERROR: Cannot calculate returns - price DataFrame is empty")
        return pd.DataFrame()
    
    if return_type == 'daily':
        returns = prices.pct_change()
    elif return_type == 'weekly':
        prices_weekly = prices.resample('W').last()
        returns = prices_weekly.pct_change()
    elif return_type == 'monthly':
        prices_monthly = prices.resample('M').last()
        returns = prices_monthly.pct_change()
    else:
        raise ValueError("return_type must be 'daily', 'weekly', or 'monthly'")
    
    # Remove first row with NaN values
    returns = returns.dropna(how='all')
    
    if returns.empty:
        print("ERROR: Returns DataFrame is empty after calculation")
        return pd.DataFrame()
    
    print(f"✓ Calculated {len(returns)} {return_type} return observations for {len(returns.columns)} stocks")
    
    return returns

