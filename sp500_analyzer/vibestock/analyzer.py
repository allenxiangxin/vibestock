"""
Analyzer Module

Statistical analysis functions for return distributions.
"""

import pandas as pd
import numpy as np
from scipy import stats


def analyze_return_distribution(returns):
    """
    Analyze and print statistics for return distributions.
    
    Args:
        returns (pd.DataFrame): DataFrame with returns
        
    Returns:
        pd.DataFrame: Summary statistics for each stock
    """
    print("\n" + "="*80)
    print("RETURN DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Check if returns DataFrame is empty
    if returns.empty or len(returns.columns) == 0:
        print("\nERROR: Returns DataFrame is empty. Cannot perform analysis.")
        print("Please check that:")
        print("  - Stock data was successfully downloaded")
        print("  - Date range is valid")
        print("  - Network connection is working")
        return pd.DataFrame()
    
    print(f"\nAnalyzing {len(returns.columns)} stocks over {len(returns)} periods")
    
    # Calculate statistics for each stock
    stats_dict = {
        'Mean': returns.mean(),
        'Median': returns.median(),
        'Std Dev': returns.std(),
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis(),
        'Min': returns.min(),
        'Max': returns.max(),
        '5th Percentile': returns.quantile(0.05),
        '95th Percentile': returns.quantile(0.95),
    }
    
    summary = pd.DataFrame(stats_dict).T
    
    # Calculate aggregate statistics across all stocks
    _print_aggregate_stats(returns)
    
    # Print top and bottom performers (only if we have valid data)
    if not summary.empty and summary.loc['Mean'].notna().any():
        _print_top_performers(summary)
        _print_worst_performers(summary)
        _print_most_volatile(summary)
    
    return summary


def _print_aggregate_stats(returns):
    """Print aggregate statistics across all stocks."""
    print("\nAGGREGATE STATISTICS (across all stocks):")
    print("-" * 80)
    all_returns = returns.values.flatten()
    all_returns = all_returns[~np.isnan(all_returns)]
    
    # Check if we have valid data
    if len(all_returns) == 0:
        print("ERROR: No valid return data available. All returns are NaN.")
        print("This may indicate:")
        print("  - No price data was downloaded")
        print("  - All downloaded data contains missing values")
        print("  - Date range is too short (need at least 2 data points)")
        return
    
    print(f"Total observations: {len(all_returns):,}")
    print(f"Mean return: {np.mean(all_returns):.6f} ({np.mean(all_returns)*100:.4f}%)")
    print(f"Median return: {np.median(all_returns):.6f} ({np.median(all_returns)*100:.4f}%)")
    print(f"Std deviation: {np.std(all_returns):.6f} ({np.std(all_returns)*100:.4f}%)")
    print(f"Skewness: {stats.skew(all_returns):.4f}")
    print(f"Kurtosis: {stats.kurtosis(all_returns):.4f}")
    print(f"Min return: {np.min(all_returns):.6f} ({np.min(all_returns)*100:.2f}%)")
    print(f"Max return: {np.max(all_returns):.6f} ({np.max(all_returns)*100:.2f}%)")
    print(f"5th percentile: {np.percentile(all_returns, 5):.6f} ({np.percentile(all_returns, 5)*100:.2f}%)")
    print(f"95th percentile: {np.percentile(all_returns, 95):.6f} ({np.percentile(all_returns, 95)*100:.2f}%)")
    
    # Test for normality (only if we have enough data)
    if len(all_returns) >= 3:
        sample_size = min(5000, len(all_returns))
        shapiro_stat, shapiro_p = stats.shapiro(all_returns[:sample_size])
        print(f"\nShapiro-Wilk normality test (p-value): {shapiro_p:.6f}")
        if shapiro_p < 0.05:
            print("Returns are NOT normally distributed (reject null hypothesis)")
        else:
            print("Returns appear normally distributed (fail to reject null hypothesis)")
    else:
        print("\nInsufficient data for normality test (need at least 3 observations)")


def _print_top_performers(summary):
    """Print top 10 stocks by mean return."""
    print("\n" + "="*80)
    print("TOP 10 STOCKS BY MEAN RETURN:")
    print("-" * 80)
    top_performers = summary.loc['Mean'].sort_values(ascending=False).head(10)
    for i, (ticker, mean_return) in enumerate(top_performers.items(), 1):
        print(f"{i:2d}. {ticker:6s}: {mean_return*100:7.4f}% (Std: {summary.loc['Std Dev', ticker]*100:.4f}%)")


def _print_worst_performers(summary):
    """Print bottom 10 stocks by mean return."""
    print("\n" + "="*80)
    print("BOTTOM 10 STOCKS BY MEAN RETURN:")
    print("-" * 80)
    worst_performers = summary.loc['Mean'].sort_values(ascending=True).head(10)
    for i, (ticker, mean_return) in enumerate(worst_performers.items(), 1):
        print(f"{i:2d}. {ticker:6s}: {mean_return*100:7.4f}% (Std: {summary.loc['Std Dev', ticker]*100:.4f}%)")


def _print_most_volatile(summary):
    """Print top 10 most volatile stocks."""
    print("\n" + "="*80)
    print("TOP 10 MOST VOLATILE STOCKS:")
    print("-" * 80)
    most_volatile = summary.loc['Std Dev'].sort_values(ascending=False).head(10)
    for i, (ticker, std) in enumerate(most_volatile.items(), 1):
        print(f"{i:2d}. {ticker:6s}: {std*100:7.4f}% (Mean: {summary.loc['Mean', ticker]*100:.4f}%)")


def calculate_sharpe_ratios(summary_stats, risk_free_rate=0.0):
    """
    Calculate Sharpe ratios for all stocks.
    
    Args:
        summary_stats (pd.DataFrame): Summary statistics DataFrame
        risk_free_rate (float): Risk-free rate (default: 0.0)
        
    Returns:
        pd.Series: Sharpe ratios for each stock
    """
    sharpe_ratios = (summary_stats.loc['Mean'] - risk_free_rate) / summary_stats.loc['Std Dev']
    return sharpe_ratios

