#!/usr/bin/env python3
"""
Main Script - S&P 500 Stock Return Distribution Analysis

Entry point for running the complete S&P 500 analysis.
"""

import argparse
from datetime import datetime, timedelta
from vibestock import (
    get_sp500_tickers,
    calculate_returns,
    analyze_return_distribution,
    create_visualizations,
    save_results,
    get_or_download_prices,
    clear_cache
)


def main():
    """
    Main function to run the S&P 500 return analysis.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='S&P 500 Stock Return Distribution Analysis')
    parser.add_argument('--force-download', action='store_true', 
                        help='Force re-download of data even if cache is valid')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear cached data and exit')
    parser.add_argument('--days', type=int, default=365*2,
                        help='Number of days of historical data (default: 730 = 2 years)')
    parser.add_argument('--return-type', choices=['daily', 'weekly', 'monthly'],
                        default='daily', help='Type of return calculation (default: daily)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of stocks to download per batch (default: 100)')
    parser.add_argument('--delay', type=float, default=12.0,
                        help='Delay in seconds between API calls (default: 12.0 for free tier = 5 calls/min)')
    args = parser.parse_args()
    
    # Handle clear cache command
    if args.clear_cache:
        clear_cache()
        return
    
    print("\n" + "="*80)
    print("S&P 500 STOCK RETURN DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Configuration
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    return_type = args.return_type
    
    print(f"\nConfiguration:")
    print(f"  Start date: {start_date}")
    print(f"  End date: {end_date}")
    print(f"  Return type: {return_type}")
    print(f"  Force download: {args.force_download}")
    
    try:
        # Step 1: Get S&P 500 tickers
        tickers = get_sp500_tickers()
        if not tickers:
            print("\nERROR: Failed to fetch S&P 500 tickers. Exiting.")
            return
        
        # Step 2: Get stock data (from cache or download)
        prices = get_or_download_prices(tickers, start_date, end_date, 
                                       force_download=args.force_download,
                                       batch_size=args.batch_size,
                                       delay=args.delay)
        if prices.empty:
            print("\nERROR: Failed to get stock data. Exiting.")
            return
        
        # Step 3: Calculate returns
        returns = calculate_returns(prices, return_type=return_type)
        if returns.empty:
            print("\nERROR: Failed to calculate returns. Exiting.")
            return
        
        # Step 4: Analyze distribution
        summary_stats = analyze_return_distribution(returns)
        if summary_stats.empty:
            print("\nERROR: Analysis failed. Exiting.")
            return
        
        # Step 5: Create visualizations
        create_visualizations(returns, summary_stats)
        
        # Step 6: Save results
        save_results(summary_stats, returns)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nResults saved in 'output/' directory:")
        print("  - summary_statistics.csv")
        print("  - returns_data.csv")
        print("  - top_performers.csv")
        print("  - return_distribution_analysis.png")
        print("  - cumulative_returns.png")
        print("  - correlation_matrix.png")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

