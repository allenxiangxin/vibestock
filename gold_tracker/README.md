# Gold Price Tracker with MACD Indicator

A Python script that monitors gold prices (via GLD ETF) using the MACD technical indicator and sends desktop notifications when buy signals are detected.

## Overview

This tool checks gold prices once per day, calculates the MACD (Moving Average Convergence Divergence) indicator, and alerts you via desktop notification when a buy signal is detected (MACD crosses above the signal line).

## Features

- **MACD Technical Analysis**: Uses 12/26/9 standard MACD parameters
- **Desktop Notifications**: Native OS notifications (macOS, Windows, Linux)
- **Daily Monitoring**: Checks once per day automatically
- **Daemon Mode**: Runs continuously in the background
- **Test Mode**: Verify notifications work before going live
- **Activity Logging**: Tracks all checks and signals
- **Professional Data**: Uses Polygon.io API for reliable GLD ETF data

## Quick Start

### Prerequisites

1. Python 3.8+ with conda environment
2. Polygon.io API key (free tier works great!)
3. API key configured in `.env` file at project root

### Installation

```bash
# Activate environment (from project root)
conda activate vibestock

# Install dependencies (if not already done)
pip install -r ../requirements.txt
```

### Basic Usage

```bash
# Test notifications (recommended first step)
python gold_tracker.py --test

# Check gold price once
python gold_tracker.py

# Run continuously (checks once per day)
python gold_tracker.py --daemon
```

## Running in Background

### macOS (Recommended: launchd)

Create a launch agent to run automatically:

```bash
# 1. Create plist file
nano ~/Library/LaunchAgents/com.goldtracker.daily.plist
```

Paste this content (adjust paths if needed):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.goldtracker.daily</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/anaconda3/envs/vibestock/bin/python</string>
        <string>/Users/YOUR_USERNAME/path/to/vibestock/gold_tracker/gold_tracker.py</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/Users/YOUR_USERNAME/path/to/vibestock/gold_tracker/gold_tracker_out.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/YOUR_USERNAME/path/to/vibestock/gold_tracker/gold_tracker_err.log</string>
    <key>WorkingDirectory</key>
    <string>/Users/YOUR_USERNAME/path/to/vibestock/gold_tracker</string>
</dict>
</plist>
```

```bash
# 2. Load the launch agent
launchctl load ~/Library/LaunchAgents/com.goldtracker.daily.plist

# 3. Verify it's running
launchctl list | grep goldtracker
```

**Management commands:**
```bash
# Stop
launchctl unload ~/Library/LaunchAgents/com.goldtracker.daily.plist

# Restart
launchctl unload ~/Library/LaunchAgents/com.goldtracker.daily.plist
launchctl load ~/Library/LaunchAgents/com.goldtracker.daily.plist

# View logs
tail -f gold_tracker_out.log
```

### Alternative: Simple Background Run

```bash
# Run in background with nohup
nohup python gold_tracker.py --daemon > gold_tracker.log 2>&1 &

# View logs
tail -f gold_tracker.log

# Stop (find PID first)
ps aux | grep gold_tracker
kill <PID>
```

## How It Works

### MACD Indicator

The MACD (Moving Average Convergence Divergence) is a momentum indicator:

- **MACD Line**: 12-day EMA - 26-day EMA
- **Signal Line**: 9-day EMA of MACD line
- **Buy Signal**: When MACD crosses above the signal line

### Data Source

Uses **GLD (SPDR Gold Shares ETF)** as a proxy for gold spot prices:
- Highly correlated with gold spot prices (X:XAUUSD)
- Backed by physical gold bullion
- Available on Polygon.io free tier
- Liquid and reliable data

### Check Frequency

- Downloads 90 days of historical GLD data
- Calculates MACD indicators
- Detects crossover signals
- Sends notification if buy signal detected
- Logs all activity

## Configuration

### API Key Setup

Create `.env` file in **project root**:

```bash
POLYGON_API_KEY=your_actual_api_key_here
```

See `../../docs/POLYGON_SETUP.md` for detailed setup instructions.

### Adjust Check Time

Edit the plist file `Hour` and `Minute` values:

```xml
<key>Hour</key>
<integer>14</integer>  <!-- 2 PM -->
<key>Minute</key>
<integer>30</integer>  <!-- 30 minutes -->
```

Market close is 4 PM ET, so checking around 5-6 PM ET ensures you have the latest data.

### Customize MACD Parameters

Edit `gold_tracker.py`:

```python
def calculate_macd(prices, fast=12, slow=26, signal=9):
    # Adjust parameters here
```

Common alternatives:
- Shorter-term: 5/13/5
- Longer-term: 19/39/9

## Output

### Logs

Activity is logged to `gold_tracker_log.txt`:

```
2024-11-08 09:00:15 - Starting gold price check...
2024-11-08 09:00:17 - Downloaded 90 days of GLD data
2024-11-08 09:00:17 - MACD calculated: -0.45, Signal: -0.32
2024-11-08 09:00:17 - ⚠️  BUY SIGNAL DETECTED! MACD crossed above signal line
2024-11-08 09:00:17 - Notification sent successfully
```

### Notifications

Desktop notifications show:
- **Title**: "Gold MACD Buy Signal! 🚀" or "Gold Price Check"
- **Message**: Current price, MACD values, and signal status
- **Sound**: Alert sound (macOS: "Glass")

## Platform Support

| Platform | Notification Method | Supported |
|----------|-------------------|-----------|
| **macOS** | `osascript` (AppleScript) | ✅ Yes |
| **Windows** | `win10toast` | ✅ Yes |
| **Linux** | `notify-send` | ✅ Yes |

## Troubleshooting

### No notifications appearing

**macOS:**
- Check System Preferences → Notifications → Script Editor (allow notifications)
- Test with: `osascript -e 'display notification "Test" with title "Test"'`

**Windows:**
- Ensure Windows notification settings allow Python notifications
- Check Windows 10 Toast is installed: `pip show win10toast`

**Linux:**
- Install libnotify: `sudo apt install libnotify-bin`
- Test with: `notify-send "Test" "Message"`

### "No data returned for X:XAUUSD"

This is expected! The script now uses **GLD** instead of `X:XAUUSD` because:
- X:XAUUSD requires forex data (not available on free tier)
- GLD is a gold ETF that tracks spot gold prices closely
- GLD works perfectly on the free tier

### API rate limits

Free tier (5 calls/min) is more than sufficient:
- Only makes 1 API call per day
- No rate limit issues

## Documentation

See **[docs/GOLD_TRACKER_GUIDE.md](docs/GOLD_TRACKER_GUIDE.md)** for comprehensive guide including:
- Detailed MACD explanation
- Advanced scheduling options
- Multiple notification methods
- Customization examples

## Why GLD Instead of Gold Spot?

**GLD (SPDR Gold Shares)** is an excellent proxy for gold spot prices:

1. **Physical Backing**: Each share represents ~0.1 oz of gold bullion
2. **High Correlation**: Tracks gold spot prices very closely (>0.99 correlation)
3. **Arbitrage**: Market makers keep GLD aligned with gold prices
4. **Liquidity**: High trading volume ensures accurate pricing
5. **Availability**: Available on Polygon.io free tier

For MACD analysis (which looks at price momentum), GLD is effectively equivalent to gold spot prices.

## Related Tools

- **[../sp500_analyzer/](../sp500_analyzer/)** - S&P 500 return distribution analyzer
- **[../docs/](../docs/)** - Shared documentation

---

**Part of the vibestock project** - See [main README](../README.md) for full project documentation.

