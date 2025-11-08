#!/usr/bin/env python3
"""
Gold Price MACD Tracker

Tracks gold price using MACD indicator and sends desktop notifications
when a buy signal is detected.

Usage:
    python gold_tracker.py              # Check once and exit
    python gold_tracker.py --daemon     # Run continuously, check daily
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from polygon import RESTClient
from dotenv import load_dotenv

# Platform-specific notification imports
try:
    if sys.platform == 'darwin':  # macOS
        import subprocess
    elif sys.platform == 'win32':  # Windows
        from win10toast import ToastNotifier
    else:  # Linux
        import subprocess
except ImportError:
    print("Warning: Some notification dependencies may not be available")

# Load environment variables
load_dotenv()


def get_polygon_client():
    """Get Polygon.io API client."""
    api_key = os.getenv('POLYGON_API_KEY')
    
    if not api_key or api_key == 'your_api_key_here':
        print("ERROR: Polygon.io API key not found!")
        print("Please set POLYGON_API_KEY in .env file")
        sys.exit(1)
    
    return RESTClient(api_key)


def download_gold_data(days=90):
    """
    Download gold price data.
    
    Args:
        days (int): Number of days of historical data to fetch
        
    Returns:
        pd.DataFrame: DataFrame with date index and close prices
    """
    print(f"\n{'='*70}")
    print("📊 GOLD PRICE TRACKER - MACD ANALYSIS")
    print(f"{'='*70}")
    print(f"Fetching {days} days of gold price data...")
    
    client = get_polygon_client()
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Using GLD (SPDR Gold Shares ETF) - most reliable for free tier
    # GLD tracks gold price and is available on all Polygon.io plans
    ticker = "GLD"  # Gold ETF
    
    try:
        # Get aggregates for gold
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start_date,
            to=end_date,
            limit=5000
        )
        
        if not aggs or len(aggs) == 0:
            print(f"ERROR: No data returned for {ticker}")
            return None
        
        # Convert to DataFrame
        dates = [datetime.fromtimestamp(agg.timestamp / 1000) for agg in aggs]
        closes = [agg.close for agg in aggs]
        volumes = [agg.volume for agg in aggs]
        
        df = pd.DataFrame({
            'close': closes,
            'volume': volumes
        }, index=dates)
        
        df.index.name = 'date'
        
        print(f"✓ Downloaded {len(df)} days of GLD (Gold ETF) data")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Latest GLD price: ${df['close'].iloc[-1]:.2f}")
        
        return df
        
    except Exception as e:
        print(f"ERROR downloading gold data: {e}")
        return None


def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Calculate MACD indicator.
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' column
        fast (int): Fast EMA period (default: 12)
        slow (int): Slow EMA period (default: 26)
        signal (int): Signal line period (default: 9)
        
    Returns:
        pd.DataFrame: DataFrame with MACD, signal, and histogram columns
    """
    print("\nCalculating MACD indicator...")
    
    # Calculate EMAs
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    # Add to dataframe
    df['macd'] = macd_line
    df['signal'] = signal_line
    df['histogram'] = histogram
    
    print(f"✓ MACD calculated (Fast: {fast}, Slow: {slow}, Signal: {signal})")
    
    return df


def detect_buy_signal(df):
    """
    Detect MACD buy signal (bullish crossover).
    
    A buy signal occurs when MACD line crosses above the signal line.
    
    Args:
        df (pd.DataFrame): DataFrame with MACD indicators
        
    Returns:
        tuple: (has_signal, signal_info)
    """
    print("\nAnalyzing MACD signals...")
    
    if len(df) < 2:
        print("⚠ Not enough data to detect signals")
        return False, None
    
    # Get last two days
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    # Check for bullish crossover (MACD crosses above signal)
    macd_cross_above = (previous['macd'] <= previous['signal']) and (current['macd'] > current['signal'])
    
    # Additional confirmation: histogram turning positive
    histogram_positive = current['histogram'] > 0
    histogram_increasing = current['histogram'] > previous['histogram']
    
    signal_info = {
        'date': current.name,
        'price': current['close'],
        'macd': current['macd'],
        'signal': current['signal'],
        'histogram': current['histogram'],
        'crossover': macd_cross_above,
        'histogram_positive': histogram_positive,
        'histogram_increasing': histogram_increasing
    }
    
    # Buy signal: MACD crosses above signal line
    has_buy_signal = macd_cross_above
    
    print(f"\nCurrent Status:")
    print(f"  Date: {current.name.strftime('%Y-%m-%d')}")
    print(f"  GLD Price: ${current['close']:.2f}")
    print(f"  MACD: {current['macd']:.4f}")
    print(f"  Signal: {current['signal']:.4f}")
    print(f"  Histogram: {current['histogram']:.4f}")
    print(f"  MACD Position: {'ABOVE' if current['macd'] > current['signal'] else 'BELOW'} signal line")
    
    if has_buy_signal:
        print(f"\n🎯 BUY SIGNAL DETECTED!")
        print(f"  MACD crossed ABOVE signal line")
        print(f"  Previous MACD: {previous['macd']:.4f} <= Signal: {previous['signal']:.4f}")
        print(f"  Current MACD: {current['macd']:.4f} > Signal: {current['signal']:.4f}")
    else:
        print(f"\n📊 No buy signal detected")
        if current['macd'] > current['signal']:
            print(f"  MACD is above signal (already in uptrend)")
        else:
            print(f"  MACD is below signal (bearish)")
    
    return has_buy_signal, signal_info


def send_notification(title, message):
    """
    Send desktop notification.
    
    Args:
        title (str): Notification title
        message (str): Notification message
    """
    print(f"\n📢 Sending notification...")
    print(f"  Title: {title}")
    print(f"  Message: {message}")
    
    try:
        if sys.platform == 'darwin':  # macOS
            # Use osascript to send notification
            script = f'display notification "{message}" with title "{title}" sound name "Glass"'
            subprocess.run(['osascript', '-e', script], check=True)
            print("✓ Notification sent (macOS)")
            
        elif sys.platform == 'win32':  # Windows
            toaster = ToastNotifier()
            toaster.show_toast(
                title,
                message,
                icon_path=None,
                duration=10,
                threaded=True
            )
            print("✓ Notification sent (Windows)")
            
        else:  # Linux
            subprocess.run([
                'notify-send',
                title,
                message,
                '--urgency=normal',
                '--icon=dialog-information'
            ], check=True)
            print("✓ Notification sent (Linux)")
            
    except Exception as e:
        print(f"⚠ Failed to send notification: {e}")
        print("  Notification content:")
        print(f"    {title}")
        print(f"    {message}")
    
    # Bell sound
    print("\a")


def save_log(signal_info, has_signal):
    """Save signal detection to log file."""
    log_file = 'gold_tracker_log.txt'
    
    with open(log_file, 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        status = "BUY SIGNAL" if has_signal else "NO SIGNAL"
        
        f.write(f"\n{'='*70}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Status: {status}\n")
        f.write(f"GLD Price: ${signal_info['price']:.2f}\n")
        f.write(f"MACD: {signal_info['macd']:.4f}\n")
        f.write(f"Signal: {signal_info['signal']:.4f}\n")
        f.write(f"Histogram: {signal_info['histogram']:.4f}\n")
        f.write(f"{'='*70}\n")
    
    print(f"\n✓ Log saved to {log_file}")


def check_gold_signal():
    """Main function to check for gold buy signals."""
    
    # Download gold data
    df = download_gold_data(days=90)  # Need enough data for MACD calculation
    
    if df is None or df.empty:
        print("ERROR: Failed to download gold data")
        return False
    
    # Calculate MACD
    df = calculate_macd(df)
    
    # Detect buy signal
    has_signal, signal_info = detect_buy_signal(df)
    
    # Save to log
    save_log(signal_info, has_signal)
    
    # Send notification if buy signal detected
    if has_signal:
        title = "🎯 Gold Buy Signal!"
        message = f"MACD crossed above signal line. Price: ${signal_info['price']:.2f}"
        send_notification(title, message)
        
        # Also send a system sound
        print("\a")  # Bell sound
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}\n")
    
    return has_signal


def daemon_mode():
    """Run in daemon mode - check once per day."""
    print("\n🔄 Running in DAEMON MODE")
    print("Will check gold price once per day")
    print("Press Ctrl+C to stop\n")
    
    while True:
        try:
            # Run the check
            check_gold_signal()
            
            # Calculate time until next check (24 hours)
            next_check = datetime.now() + timedelta(days=1)
            print(f"\n⏰ Next check: {next_check.strftime('%Y-%m-%d %H:%M:%S')}")
            print("Sleeping for 24 hours...\n")
            
            # Sleep for 24 hours (86400 seconds)
            time.sleep(86400)
            
        except KeyboardInterrupt:
            print("\n\n👋 Stopping daemon mode...")
            break
        except Exception as e:
            print(f"\n⚠ Error in daemon mode: {e}")
            print("Waiting 1 hour before retry...\n")
            time.sleep(3600)  # Wait 1 hour on error


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Gold Price MACD Tracker - Desktop Notifications'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run in daemon mode (check once per day continuously)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test notification system'
    )
    
    args = parser.parse_args()
    
    if args.test:
        print("Testing notification system...")
        send_notification(
            "Gold Tracker Test",
            "If you see this, notifications are working!"
        )
        return
    
    if args.daemon:
        daemon_mode()
    else:
        # Single check mode
        check_gold_signal()


if __name__ == "__main__":
    main()

