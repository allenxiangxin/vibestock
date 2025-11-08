# Polygon.io Setup Guide

This project now uses **Polygon.io** instead of Yahoo Finance for reliable, unlimited stock data access.

## 🚀 Quick Setup (5 minutes)

### Step 1: Get Your Polygon.io API Key

1. **Sign up** for a free account at https://polygon.io/
   - Free tier includes:
     - ✅ 5 API calls per minute
     - ✅ Unlimited daily requests
     - ✅ 2 years of historical data
     - ✅ Delayed market data (15 minutes)

2. **Verify your email** and log in

3. **Get your API key** from the dashboard
   - Click on your profile/dashboard
   - Copy your API key (looks like: `abc123def456...`)

### Step 2: Configure Your API Key

**Option A: Using .env file (Recommended)**

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` in your text editor:
   ```bash
   nano .env
   # or
   code .env
   ```

3. Replace `your_api_key_here` with your actual key:
   ```bash
   POLYGON_API_KEY=abc123def456ghi789jkl...
   ```

4. Save the file

**Option B: Using environment variable**

```bash
# Temporarily (current session only)
export POLYGON_API_KEY=your_actual_api_key_here

# Permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export POLYGON_API_KEY=your_actual_api_key_here' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Analysis

```bash
python main.py
```

That's it! Your first download will take ~10-15 minutes for 500 S&P 500 stocks (depending on your plan), then subsequent runs use cached data (30 seconds).

## 📊 Polygon.io Plans Comparison

| Plan | Cost | API Calls/Min | Best For |
|------|------|---------------|----------|
| **Free (Launchpad)** | $0/month | 5 calls/min | Hobby projects, learning |
| **Starter** | $29/month | 100 calls/min | Serious analysis |
| **Developer** | $99/month | Unlimited | Professional use |

### Download Time Estimates

For ~500 S&P 500 stocks with 2 years of data:

| Plan | Time | Notes |
|------|------|-------|
| Free (5 calls/min) | ~100 min | First download only, then cached |
| Starter (100 calls/min) | ~5 min | Much faster |
| Developer (unlimited) | ~2 min | Fastest |

**Remember**: After the first download, the data is **cached**, so subsequent runs take only 10-30 seconds!

## 🔧 Configuration Options

### Adjusting Rate Limits

If you're hitting rate limits on the free tier:

```bash
# Slower but safer (1 second between calls)
python main.py --delay 1.0

# Very conservative (2 seconds between calls)
python main.py --delay 2.0
```

If you have a paid plan:

```bash
# Faster (0.1 seconds between calls)
python main.py --delay 0.1

# No delay (unlimited plan)
python main.py --delay 0.01
```

### Custom Time Periods

```bash
# 5 years of data
python main.py --days 1825

# 1 year of data (faster)
python main.py --days 365
```

## ✅ Verify Your Setup

Test your API key with this simple script:

```python
# test_polygon.py
import os
from dotenv import load_dotenv
from polygon import RESTClient

load_dotenv()

api_key = os.getenv('POLYGON_API_KEY')

if not api_key:
    print("❌ API key not found!")
    print("Please set POLYGON_API_KEY in .env file")
else:
    print(f"✓ API key found: {api_key[:8]}...")
    
    try:
        client = RESTClient(api_key)
        # Test API call
        aggs = client.get_aggs("AAPL", 1, "day", "2024-10-01", "2024-10-30")
        if aggs:
            print(f"✓ API working! Downloaded {len(aggs)} days of AAPL data")
        else:
            print("⚠ API call succeeded but no data returned")
    except Exception as e:
        print(f"❌ API error: {e}")
```

Run it:
```bash
python test_polygon.py
```

## 🆚 Polygon.io vs Yahoo Finance

| Feature | Yahoo Finance (yfinance) | Polygon.io |
|---------|-------------------------|------------|
| **Reliability** | ⚠️ Unreliable, frequent rate limits | ✅ Very reliable |
| **Rate Limits** | Very strict, unpredictable | Clear, documented limits |
| **Cost** | Free only | Free tier + paid options |
| **Support** | None (unofficial API) | Official support |
| **Data Quality** | Good | Excellent |
| **Documentation** | Community-driven | Professional |
| **Legal** | Gray area (scraping) | Fully legal, official API |

## 🐛 Troubleshooting

### Error: "API key not found"

**Solution:**
1. Make sure `.env` file exists in project root
2. Check that `POLYGON_API_KEY=...` is set correctly
3. No spaces around the `=` sign
4. No quotes around the API key

### Error: "Rate limited" or "429 Too Many Requests"

**Solution:**
```bash
# Increase delay between calls
python main.py --delay 1.5

# For free tier, use conservative settings
python main.py --delay 2.0
```

The script will automatically retry with backoff.

### Error: "Invalid API key" or "401 Unauthorized"

**Solution:**
1. Double-check your API key in the Polygon.io dashboard
2. Make sure you copied the entire key
3. Verify your email if you just signed up

### Downloads are slow

**Normal on free tier!** Free plan allows 5 API calls/minute:
- 500 stocks = ~100 minutes first download
- But then cached forever (30 seconds for subsequent runs)

**To speed up:**
- Upgrade to Starter plan ($29/month) = 5 minutes
- Or just be patient once, then enjoy cached speeds

### No data for certain stocks

Some stocks may:
- Be newly listed (less than 2 years of data)
- Have been delisted
- Have different ticker symbols

This is normal - the script filters them out automatically.

## 💡 Tips

1. **Download once, analyze many times**
   - First run: ~10-100 minutes (depending on plan)
   - All other runs: 30 seconds (uses cache)

2. **Run overnight** if on free tier
   - Set it running before bed
   - Wake up to cached data!

3. **Upgrade strategically**
   - Start with free tier to test
   - Upgrade to Starter ($29) if you're doing this regularly
   - Developer ($99) only if you need real-time or very frequent updates

4. **Cache is your friend**
   - Cache is valid for 24 hours
   - Automatically refreshes when stale
   - No need to re-download unless you want fresh data

## 📚 Additional Resources

- **Polygon.io Documentation**: https://polygon.io/docs
- **API Reference**: https://polygon.io/docs/stocks
- **Pricing**: https://polygon.io/pricing
- **Support**: support@polygon.io

## 🎉 Success!

Once set up, your workflow is:

```bash
# First time (or when cache expires)
python main.py  # Takes 10-100 min depending on plan

# All subsequent times (using cache)
python main.py  # Takes 30 seconds ⚡
python main.py  # Takes 30 seconds ⚡
python main.py  # Takes 30 seconds ⚡
```

Enjoy reliable, unlimited stock data! 🚀

