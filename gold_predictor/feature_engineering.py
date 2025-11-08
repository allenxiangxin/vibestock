"""
Gold Predictor - Feature Engineering Module

Creates features from raw economic and market data for gold price prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_returns(df: pd.DataFrame, column: str, periods: list) -> pd.DataFrame:
    """
    Calculate returns over multiple periods.
    
    Args:
        df: DataFrame with price data
        column: Column name to calculate returns from
        periods: List of periods (in days)
        
    Returns:
        DataFrame with return columns added
    """
    result = df.copy()
    
    for period in periods:
        result[f'{column}_return_{period}d'] = (
            result[column].pct_change(periods=period) * 100
        )
    
    return result


def calculate_volatility(df: pd.DataFrame, column: str, windows: list) -> pd.DataFrame:
    """
    Calculate rolling volatility.
    
    Args:
        df: DataFrame with price data
        column: Column name to calculate volatility from
        windows: List of window sizes (in days)
        
    Returns:
        DataFrame with volatility columns added
    """
    result = df.copy()
    
    # First calculate daily returns
    daily_returns = result[column].pct_change() * 100
    
    for window in windows:
        result[f'{column}_vol_{window}d'] = (
            daily_returns.rolling(window=window).std()
        )
    
    return result


def calculate_moving_averages(df: pd.DataFrame, column: str, windows: list) -> pd.DataFrame:
    """
    Calculate moving averages and crossovers.
    
    Args:
        df: DataFrame with price data
        column: Column name
        windows: List of window sizes
        
    Returns:
        DataFrame with MA columns added
    """
    result = df.copy()
    
    for window in windows:
        result[f'{column}_ma_{window}d'] = (
            result[column].rolling(window=window).mean()
        )
    
    return result


def calculate_real_interest_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate real interest rate (nominal - inflation).
    
    Args:
        df: DataFrame with treasury_10y and cpi data
        
    Returns:
        DataFrame with real_rate column added
    """
    result = df.copy()
    
    if 'treasury_10y' in result.columns and 'cpi' in result.columns:
        # Calculate year-over-year CPI inflation rate
        result['inflation_rate'] = result['cpi'].pct_change(periods=252) * 100  # Annual
        
        # Real rate = Nominal rate - Inflation
        result['real_interest_rate'] = result['treasury_10y'] - result['inflation_rate']
    
    return result


def calculate_momentum_indicators(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Calculate momentum indicators (RSI, MACD).
    
    Args:
        df: DataFrame with price data
        column: Price column name
        
    Returns:
        DataFrame with momentum indicators added
    """
    result = df.copy()
    
    # RSI (Relative Strength Index)
    delta = result[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    result[f'{column}_rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = result[column].ewm(span=12, adjust=False).mean()
    exp2 = result[column].ewm(span=26, adjust=False).mean()
    result[f'{column}_macd'] = exp1 - exp2
    result[f'{column}_macd_signal'] = result[f'{column}_macd'].ewm(span=9, adjust=False).mean()
    result[f'{column}_macd_hist'] = result[f'{column}_macd'] - result[f'{column}_macd_signal']
    
    return result


def create_target_variables(df: pd.DataFrame, horizons: dict) -> pd.DataFrame:
    """
    Create target variables for prediction (future price direction).
    
    Args:
        df: DataFrame with gold price data
        horizons: Dict mapping term names to days (e.g., {'short': 7, 'mid': 30, 'long': 90})
        
    Returns:
        DataFrame with target columns added
    """
    result = df.copy()
    
    for term, days in horizons.items():
        # Future return
        future_return = result['close'].pct_change(periods=days).shift(-days) * 100
        
        # Binary target: 1 if price goes up, 0 if down
        result[f'target_{term}'] = (future_return > 0).astype(int)
        
        # Continuous target: actual return
        result[f'target_{term}_return'] = future_return
    
    return result


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all features for gold price prediction.
    
    Args:
        df: Raw merged data from data_fetcher
        
    Returns:
        DataFrame with all engineered features
    """
    print("🔧 Engineering features...")
    
    result = df.copy()
    
    # 1. Gold price features
    print("  • Gold price returns...")
    result = calculate_returns(result, 'close', [1, 7, 30, 90])
    
    print("  • Gold price volatility...")
    result = calculate_volatility(result, 'close', [7, 30, 90])
    
    print("  • Gold moving averages...")
    result = calculate_moving_averages(result, 'close', [7, 30, 90, 200])
    
    print("  • Gold momentum indicators...")
    result = calculate_momentum_indicators(result, 'close')
    
    # 2. USD Index features
    if 'usd_close' in result.columns:
        print("  • USD returns and volatility...")
        result = calculate_returns(result, 'usd_close', [7, 30, 90])
        result = calculate_volatility(result, 'usd_close', [30])
    
    # 3. Real interest rate (most important!)
    if 'treasury_10y' in result.columns and 'cpi' in result.columns:
        print("  • Real interest rate...")
        result = calculate_real_interest_rate(result)
    
    # 4. Rate of change for economic indicators
    if 'fed_funds_rate' in result.columns:
        print("  • Fed funds rate changes...")
        result['fed_funds_change'] = result['fed_funds_rate'].diff()
    
    if 'inflation_expectations' in result.columns:
        print("  • Inflation expectation changes...")
        result['inflation_exp_change'] = result['inflation_expectations'].diff()
    
    # 5. VIX (geopolitical stress proxy)
    if 'vix' in result.columns:
        print("  • VIX features...")
        result = calculate_moving_averages(result, 'vix', [7, 30])
        result['vix_change'] = result['vix'].diff()
    
    # 6. Create target variables
    print("  • Target variables (short/mid/long term)...")
    horizons = {
        'short': 7,    # 1 week
        'mid': 30,     # 1 month
        'long': 90     # 3 months
    }
    result = create_target_variables(result, horizons)
    
    # Drop rows with NaN (from rolling calculations and future targets)
    initial_len = len(result)
    result = result.dropna()
    dropped = initial_len - len(result)
    
    print(f"\n✅ Feature engineering complete!")
    print(f"   Total features: {len(result.columns)}")
    print(f"   Observations: {len(result)} (dropped {dropped} incomplete rows)")
    print(f"   Features available: {list(result.columns)}\n")
    
    return result


def get_feature_importance_names() -> dict:
    """
    Return human-readable names for feature importance display.
    
    Returns:
        Dict mapping feature column names to readable descriptions
    """
    return {
        'real_interest_rate': 'Real Interest Rate (10Y-Inflation)',
        'usd_close_return_30d': 'USD Index 30-day Return',
        'inflation_expectations': 'Inflation Expectations (5Y)',
        'fed_funds_rate': 'Federal Funds Rate',
        'fed_funds_change': 'Fed Funds Rate Change',
        'vix': 'VIX (Fear Index)',
        'vix_change': 'VIX Change',
        'close_vol_30d': 'Gold 30-day Volatility',
        'close_return_30d': 'Gold 30-day Return',
        'close_rsi': 'Gold RSI',
        'close_macd_hist': 'Gold MACD Histogram',
        'inflation_rate': 'CPI Inflation Rate',
        'inflation_exp_change': 'Inflation Expectation Change',
    }


def select_features_for_model(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select relevant features for modeling.
    
    Args:
        df: DataFrame with all features
        target: Target column name
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Define feature groups
    gold_features = [
        'close_return_7d', 'close_return_30d',
        'close_vol_30d', 'close_rsi', 'close_macd_hist'
    ]
    
    usd_features = [
        'usd_close_return_30d', 'usd_close_vol_30d'
    ]
    
    econ_features = [
        'real_interest_rate', 'fed_funds_rate', 'fed_funds_change',
        'inflation_rate', 'inflation_expectations', 'inflation_exp_change'
    ]
    
    risk_features = [
        'vix', 'vix_change'
    ]
    
    # Combine all available features
    all_feature_cols = gold_features + usd_features + econ_features + risk_features
    
    # Select only columns that exist in the DataFrame
    available_features = [col for col in all_feature_cols if col in df.columns]
    
    print(f"📊 Selected {len(available_features)} features for modeling:")
    for feat in available_features:
        print(f"   • {feat}")
    
    X = df[available_features].copy()
    y = df[target].copy()
    
    return X, y


if __name__ == "__main__":
    """Test feature engineering."""
    print("Testing Feature Engineering Module\n")
    
    # Load sample data
    try:
        df = pd.read_csv('data/gold_predictor_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Loaded {len(df)} rows of data\n")
    except FileNotFoundError:
        print("ERROR: Run data_fetcher.py first to generate sample data")
        import sys
        sys.exit(1)
    
    # Engineer features
    df_features = engineer_all_features(df)
    
    # Show sample
    print("\n" + "="*80)
    print("SAMPLE FEATURES (last 5 rows):")
    print("="*80)
    print(df_features[['date', 'close', 'real_interest_rate', 'usd_close_return_30d', 
                       'target_short', 'target_mid', 'target_long']].tail())
    
    # Save
    output_file = 'data/gold_features.csv'
    df_features.to_csv(output_file, index=False)
    print(f"\n✅ Features saved to: {output_file}")

