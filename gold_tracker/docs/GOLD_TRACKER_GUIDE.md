# Gold Price MACD Tracker Guide

## 🎯 What This Does

Tracks **GLD (SPDR Gold Shares ETF)** price using the MACD (Moving Average Convergence Divergence) indicator and sends you a **desktop notification** when a buy signal is detected.

**Note**: GLD is a gold ETF that tracks physical gold prices. It's available on Polygon.io free tier and closely follows gold spot prices.

## 📊 MACD Indicator Explained

**MACD** is a popular technical indicator that shows the relationship between two moving averages:

- **MACD Line** = 12-day EMA - 26-day EMA
- **Signal Line** = 9-day EMA of MACD line
- **Buy Signal** = MACD line crosses ABOVE signal line (bullish crossover)

## 🚀 Quick Start

### 1. Test Notification System

First, make sure notifications work on your computer:

```bash
python gold_tracker.py --test
```

You should see a notification appear on your screen!

### 2. Run One-Time Check

Check gold price right now:

```bash
python gold_tracker.py
```

This will:
- Download 90 days of gold price data
- Calculate MACD indicator
- Check for buy signal
- Send notification if signal detected
- Save results to log file

### 3. Run Daily Checks (Daemon Mode)

Run continuously and check once per day:

```bash
python gold_tracker.py --daemon
```

This will:
- Check gold price immediately
- Wait 24 hours
- Check again
- Repeat forever (until you stop it with Ctrl+C)

## 📱 Notification Examples

### When Buy Signal Detected:
```
🎯 Gold Buy Signal!
MACD crossed above signal line. Price: $2045.30
```

### When No Signal:
No notification is sent, but results are logged to `gold_tracker_log.txt`

## 📋 Output Example

```
======================================================================
📊 GOLD PRICE TRACKER - MACD ANALYSIS
======================================================================
Fetching 90 days of gold price data...
✓ Downloaded 89 days of gold price data
  Date range: 2024-08-02 to 2024-10-30
  Latest price: $2045.30

Calculating MACD indicator...
✓ MACD calculated (Fast: 12, Slow: 26, Signal: 9)

Analyzing MACD signals...

Current Status:
  Date: 2024-10-30
  Gold Price: $2045.30
  MACD: 15.2341
  Signal: 12.5678
  Histogram: 2.6663
  MACD Position: ABOVE signal line

🎯 BUY SIGNAL DETECTED!
  MACD crossed ABOVE signal line
  Previous MACD: 11.2341 <= Signal: 12.5678
  Current MACD: 15.2341 > Signal: 12.5678

📢 Sending notification...
✓ Notification sent (macOS)

✓ Log saved to gold_tracker_log.txt

======================================================================
Analysis complete!
======================================================================
```

## 🔔 Setting Up Daily Checks

### Option A: Manual (Recommended for testing)

Run in daemon mode and keep terminal open:

```bash
python gold_tracker.py --daemon
```

### Option B: Using Cron (macOS/Linux)

Set up a cron job to run daily at 9 AM:

```bash
# Open crontab editor
crontab -e

# Add this line (runs at 9 AM every day)
0 9 * * * cd /Users/xxiang/MyGitHub/Gmail/vibestock && /Users/xxiang/anaconda3/envs/vibestock/bin/python gold_tracker.py >> gold_tracker_cron.log 2>&1
```

### Option C: Using Task Scheduler (Windows)

1. Open Task Scheduler
2. Create Basic Task
3. Trigger: Daily at 9:00 AM
4. Action: Start a program
   - Program: `python`
   - Arguments: `gold_tracker.py`
   - Start in: `C:\path\to\vibestock`

### Option D: Using launchd (macOS - Best Option)

Create a launch agent that runs daily:

```bash
# Create the plist file
nano ~/Library/LaunchAgents/com.goldtracker.daily.plist
```

Add this content:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.goldtracker.daily</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/xxiang/anaconda3/envs/vibestock/bin/python</string>
        <string>/Users/xxiang/MyGitHub/Gmail/vibestock/gold_tracker.py</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/Users/xxiang/MyGitHub/Gmail/vibestock/gold_tracker_out.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/xxiang/MyGitHub/Gmail/vibestock/gold_tracker_err.log</string>
    <key>WorkingDirectory</key>
    <string>/Users/xxiang/MyGitHub/Gmail/vibestock</string>
</dict>
</plist>
```

Load the launch agent:

```bash
launchctl load ~/Library/LaunchAgents/com.goldtracker.daily.plist
```

## 📊 Understanding the Output

### MACD Metrics

- **MACD Line**: Difference between fast and slow EMAs
  - Positive = Short-term trend > Long-term trend (bullish)
  - Negative = Short-term trend < Long-term trend (bearish)

- **Signal Line**: 9-day EMA of MACD line (smoothed)

- **Histogram**: MACD - Signal
  - Positive = MACD above signal (bullish momentum)
  - Negative = MACD below signal (bearish momentum)
  - Increasing = Momentum strengthening
  - Decreasing = Momentum weakening

### Buy Signal Criteria

✅ **Buy Signal**: MACD line crosses ABOVE signal line
- Previous day: MACD <= Signal
- Current day: MACD > Signal
- Indicates start of uptrend

❌ **Not a Buy Signal**:
- MACD already above signal (already in uptrend)
- MACD below signal (bearish)

## 📁 Log File

All checks are saved to `gold_tracker_log.txt`:

```
======================================================================
Timestamp: 2024-10-30 09:00:15
Status: BUY SIGNAL
Gold Price: $2045.30
MACD: 15.2341
Signal: 12.5678
Histogram: 2.6663
======================================================================
```

## 🔧 Troubleshooting

### No Notification Appearing?

**macOS:**
- Check System Preferences → Notifications
- Make sure "Script Editor" or "Terminal" has notifications enabled
- Test with: `python gold_tracker.py --test`

**Windows:**
- Install notification library: `pip install win10toast`
- Check Windows notification settings

**Linux:**
- Install: `sudo apt-get install libnotify-bin`
- Test with: `notify-send "Test" "Hello"`

### API Errors?

Make sure your Polygon.io API key is configured:
```bash
# Check .env file
cat .env
# Should show: POLYGON_API_KEY=your_key_here
```

### Gold Data Not Available?

The script uses ticker `X:XAUUSD` (Gold/USD spot price). If this doesn't work:
- Check Polygon.io dashboard for available gold tickers
- Your plan may need forex data access

## 🎮 Command Reference

```bash
# Check once and exit
python gold_tracker.py

# Run continuously (check daily)
python gold_tracker.py --daemon

# Test notifications
python gold_tracker.py --test

# View log file
cat gold_tracker_log.txt

# Stop daemon mode
# Press Ctrl+C in the terminal
```

## 💡 Tips

1. **Test First**: Run `python gold_tracker.py --test` to verify notifications work

2. **Check Logs**: Review `gold_tracker_log.txt` to see historical signals

3. **Adjust Timing**: Modify the cron/task scheduler to run at your preferred time

4. **Multiple Checks**: Run daemon mode in background for continuous monitoring

5. **Monitor Performance**: Check the log file regularly to see signal accuracy

## 🎯 Trading Strategy Tips

**MACD Buy Signals are just indicators, not guarantees!**

Best practices:
- ✅ Confirm with other indicators (RSI, volume, support/resistance)
- ✅ Consider the overall market trend
- ✅ Use stop-loss orders
- ✅ Don't trade based on single indicator alone
- ✅ MACD works best in trending markets
- ❌ MACD gives false signals in sideways/choppy markets

## 📈 Customization

Want to modify the MACD parameters? Edit `gold_tracker.py`:

```python
# In the check_gold_signal() function, change:
df = calculate_macd(df, fast=12, slow=26, signal=9)

# To your preferred values, e.g.:
df = calculate_macd(df, fast=8, slow=21, signal=5)  # Faster signals
```

## 🆘 Support

If you need help:
1. Check the log file: `gold_tracker_log.txt`
2. Test notifications: `python gold_tracker.py --test`
3. Run in debug mode and check output
4. Verify API key is configured correctly

## 🎉 Success Checklist

- ✅ Notifications work (tested with `--test`)
- ✅ Script runs successfully once
- ✅ Log file is created
- ✅ Scheduled task/cron job is set up
- ✅ You understand MACD signals

You're all set! You'll get notified whenever gold shows a MACD buy signal! 🚀

