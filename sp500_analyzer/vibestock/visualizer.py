"""
Visualizer Module

Functions for creating visualization plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def create_visualizations(returns, summary_stats, output_dir='output'):
    """
    Create and save visualization plots.
    
    Args:
        returns (pd.DataFrame): DataFrame with returns
        summary_stats (pd.DataFrame): Summary statistics DataFrame
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS...")
    print("="*80)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create all visualizations
    _create_distribution_analysis_plot(returns, summary_stats, output_dir)
    _create_cumulative_returns_plot(returns, output_dir)
    _create_correlation_matrix_plot(returns, output_dir)
    
    print("\nAll visualizations saved to 'output/' directory")


def _create_distribution_analysis_plot(returns, summary_stats, output_dir):
    """Create multi-panel distribution analysis plot."""
    plt.rcParams['figure.figsize'] = (16, 12)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Flatten all returns
    all_returns = returns.values.flatten()
    all_returns = all_returns[~np.isnan(all_returns)]
    
    # 1. Histogram with normal curve overlay
    axes[0, 0].hist(all_returns, bins=100, density=True, alpha=0.7, color='blue', edgecolor='black')
    
    # Overlay normal distribution
    mu, sigma = all_returns.mean(), all_returns.std()
    x = np.linspace(all_returns.min(), all_returns.max(), 100)
    axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
    axes[0, 0].set_xlabel('Return')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Return Distribution (All S&P 500 Stocks)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Q-Q plot
    stats.probplot(all_returns, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Returns vs Normal Distribution)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Box plot of returns by stock (sample of 20 stocks)
    sample_size = min(20, len(returns.columns))
    sample_stocks = np.random.choice(returns.columns, sample_size, replace=False)
    returns_sample = returns[sample_stocks]
    
    axes[1, 0].boxplot([returns_sample[col].dropna() for col in returns_sample.columns],
                        labels=returns_sample.columns, vert=True)
    axes[1, 0].set_xlabel('Stock Ticker')
    axes[1, 0].set_ylabel('Return')
    axes[1, 0].set_title(f'Return Distribution by Stock (Sample of {sample_size} stocks)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Mean vs Std Dev scatter
    axes[1, 1].scatter(summary_stats.loc['Std Dev'] * 100, 
                       summary_stats.loc['Mean'] * 100,
                       alpha=0.6, s=30)
    axes[1, 1].set_xlabel('Standard Deviation (%)')
    axes[1, 1].set_ylabel('Mean Return (%)')
    axes[1, 1].set_title('Risk-Return Profile (All S&P 500 Stocks)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add diagonal line for reference (Sharpe ratio = 1, assuming 0 risk-free rate)
    xlim = axes[1, 1].get_xlim()
    ylim = axes[1, 1].get_ylim()
    max_val = min(xlim[1], ylim[1])
    axes[1, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Sharpe = 1')
    axes[1, 1].legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'return_distribution_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def _create_cumulative_returns_plot(returns, output_dir):
    """Create cumulative returns time series plot."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Calculate equal-weighted portfolio returns
    portfolio_returns = returns.mean(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    ax.plot(cumulative_returns.index, cumulative_returns.values, linewidth=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (Starting at 1)')
    ax.set_title('Cumulative Returns - Equal-Weighted S&P 500 Portfolio')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cumulative_returns.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def _create_correlation_matrix_plot(returns, output_dir):
    """Create correlation matrix heatmap."""
    # Sample of stocks for correlation matrix
    sample_size = min(30, len(returns.columns))
    sample_stocks = np.random.choice(returns.columns, sample_size, replace=False)
    corr_matrix = returns[sample_stocks].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title(f'Return Correlation Matrix (Sample of {sample_size} stocks)')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_individual_stock(returns, ticker, output_dir='output'):
    """
    Create detailed plots for an individual stock.
    
    Args:
        returns (pd.DataFrame): DataFrame with returns
        ticker (str): Ticker symbol
        output_dir (str): Directory to save plots
    """
    if ticker not in returns.columns:
        print(f"Error: {ticker} not found in returns data")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    stock_returns = returns[ticker].dropna()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Returns time series
    axes[0, 0].plot(stock_returns.index, stock_returns.values)
    axes[0, 0].set_title(f'{ticker} - Returns Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution histogram
    axes[0, 1].hist(stock_returns, bins=50, density=True, alpha=0.7, edgecolor='black')
    mu, sigma = stock_returns.mean(), stock_returns.std()
    x = np.linspace(stock_returns.min(), stock_returns.max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
    axes[0, 1].set_title(f'{ticker} - Return Distribution')
    axes[0, 1].set_xlabel('Return')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Cumulative returns
    cumulative = (1 + stock_returns).cumprod()
    axes[1, 0].plot(cumulative.index, cumulative.values)
    axes[1, 0].set_title(f'{ticker} - Cumulative Returns')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Cumulative Return')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q plot
    stats.probplot(stock_returns, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f'{ticker} - Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{ticker}_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

