"""
Gold Predictor - Data Fetcher Module

Fetches economic indicators and market data for gold price prediction.
Data sources:
- FRED API (Federal Reserve Economic Data) - Economic indicators
- Polygon.io - Gold prices (GLD ETF), USD Index (UUP)
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Try to import Polygon client
try:
    from polygon import RESTClient
except ImportError:
    print("ERROR: polygon-api-client not installed. Run: pip install polygon-api-client")
    sys.exit(1)

# Try to import FRED client
try:
    from fredapi import Fred
except ImportError:
    print("WARNING: fredapi not installed. Economic data will be limited.")
    print("Install with: pip install fredapi")
    Fred = None

# Load environment variables
load_dotenv()


def get_polygon_client():
    """Get Polygon.io API client."""
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise ValueError(
            "Polygon.io API key not found! "
            "Please set POLYGON_API_KEY in .env file"
        )
    return RESTClient(api_key)


def get_fred_client():
    """Get FRED API client."""
    if Fred is None:
        return None
    
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("WARNING: FRED_API_KEY not found in .env file")
        print("Economic indicators will be simulated/limited")
        print("Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return None
    return Fred(api_key=api_key)


def download_gold_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download gold price data (GLD ETF as proxy).
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame with gold prices
    """
    print(f"📥 Downloading gold data (GLD) from {start_date} to {end_date}...")
    
    client = get_polygon_client()
    ticker = "GLD"
    
    try:
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start_date,
            to=end_date,
            limit=50000
        )
        
        if not aggs:
            print(f"ERROR: No data returned for {ticker}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for agg in aggs:
            data.append({
                'date': datetime.fromtimestamp(agg.timestamp / 1000).date(),
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume
            })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"✅ Downloaded {len(df)} days of gold data")
        return df
        
    except Exception as e:
        print(f"ERROR downloading gold data: {e}")
        return pd.DataFrame()


def download_usd_index_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download USD Index data (UUP ETF as proxy for DXY).
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame with USD index
    """
    print(f"📥 Downloading USD Index data (UUP) from {start_date} to {end_date}...")
    
    client = get_polygon_client()
    ticker = "UUP"
    
    try:
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start_date,
            to=end_date,
            limit=50000
        )
        
        if not aggs:
            print(f"WARNING: No data returned for {ticker}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for agg in aggs:
            data.append({
                'date': datetime.fromtimestamp(agg.timestamp / 1000).date(),
                'usd_close': agg.close
            })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"✅ Downloaded {len(df)} days of USD index data")
        return df
        
    except Exception as e:
        print(f"WARNING: Error downloading USD index: {e}")
        return pd.DataFrame()


def download_fred_indicator(series_id: str, name: str, 
                            start_date: str, end_date: str,
                            fred_client) -> pd.DataFrame:
    """
    Download economic indicator from FRED.
    
    Args:
        series_id: FRED series ID
        name: Human-readable name for the indicator
        start_date: Start date
        end_date: End date
        fred_client: FRED API client
        
    Returns:
        DataFrame with indicator data
    """
    if fred_client is None:
        return pd.DataFrame()
    
    try:
        print(f"📥 Downloading {name} ({series_id})...")
        series = fred_client.get_series(series_id, start_date, end_date)
        
        df = pd.DataFrame({
            'date': series.index,
            name.lower().replace(' ', '_'): series.values
        })
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"✅ Downloaded {len(df)} observations of {name}")
        return df
        
    except Exception as e:
        print(f"WARNING: Error downloading {name}: {e}")
        return pd.DataFrame()


def fetch_all_data(lookback_years: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Fetch all required data for gold prediction.
    
    Args:
        lookback_years: Number of years of historical data to fetch
        
    Returns:
        Dictionary containing all data sources
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_years * 365)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"\n{'='*60}")
    print(f"FETCHING DATA: {start_str} to {end_str}")
    print(f"{'='*60}\n")
    
    data = {}
    
    # 1. Gold prices (required)
    data['gold'] = download_gold_data(start_str, end_str)
    
    # 2. USD Index (required)
    data['usd'] = download_usd_index_data(start_str, end_str)
    
    # 3. FRED Economic Indicators (optional but recommended)
    fred_client = get_fred_client()
    
    if fred_client:
        # Real interest rates (10-Year Treasury - Inflation)
        data['treasury_10y'] = download_fred_indicator(
            'DGS10', 'treasury_10y', start_str, end_str, fred_client
        )
        
        # Inflation (CPI)
        data['cpi'] = download_fred_indicator(
            'CPIAUCSL', 'cpi', start_str, end_str, fred_client
        )
        
        # Inflation expectations (5-Year Breakeven)
        data['inflation_exp'] = download_fred_indicator(
            'T5YIE', 'inflation_expectations', start_str, end_str, fred_client
        )
        
        # Federal Funds Rate (Central bank policy)
        data['fed_funds'] = download_fred_indicator(
            'FEDFUNDS', 'fed_funds_rate', start_str, end_str, fred_client
        )
        
        # VIX (Fear/geopolitical stress proxy)
        data['vix'] = download_fred_indicator(
            'VIXCLS', 'vix', start_str, end_str, fred_client
        )
    else:
        print("\n⚠️  FRED API not configured - using limited dataset")
        print("For better predictions, get free FRED API key:")
        print("https://fred.stlouisfed.org/docs/api/api_key.html\n")
    
    print(f"\n{'='*60}")
    print(f"DATA FETCH COMPLETE")
    print(f"{'='*60}\n")
    
    return data


def merge_all_data(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all data sources into a single DataFrame.
    
    Args:
        data: Dictionary of DataFrames from fetch_all_data()
        
    Returns:
        Merged DataFrame with all indicators
    """
    print("🔄 Merging all data sources...")
    
    if 'gold' not in data or data['gold'].empty:
        raise ValueError("Gold price data is required!")
    
    # Start with gold data
    df = data['gold'].copy()
    
    # Merge each data source
    for key, source_df in data.items():
        if key == 'gold' or source_df.empty:
            continue
        
        # Merge on date
        df = pd.merge(df, source_df, on='date', how='left')
    
    # Forward fill missing values (economic data is often monthly/quarterly)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
    
    # Drop rows with any remaining NaN
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    
    if dropped > 0:
        print(f"⚠️  Dropped {dropped} rows with missing data")
    
    print(f"✅ Merged data: {len(df)} complete observations")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Features: {list(df.columns)}\n")
    
    return df


if __name__ == "__main__":
    """Test data fetching."""
    print("Testing Gold Predictor Data Fetcher\n")
    
    # Fetch data
    data = fetch_all_data(lookback_years=2)
    
    # Merge data
    df = merge_all_data(data)
    
    print("\n" + "="*60)
    print("SAMPLE DATA (last 5 rows):")
    print("="*60)
    print(df.tail())
    
    # Save to CSV for inspection
    output_file = 'data/gold_predictor_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Sample data saved to: {output_file}")

