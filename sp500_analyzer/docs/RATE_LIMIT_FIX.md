# Yahoo Finance Rate Limiting - Solutions

## ⚠️ Issue

Yahoo Finance's free API has strict rate limits. Even with batching, you may encounter:
```
YFRateLimitError('Too Many Requests. Rate limited. Try after a while.')
```

## ✅ Updated Default Settings

I've updated the default settings to be more conservative:
- **Batch size**: 50 → **10 stocks** per batch
- **Delay**: 1.0s → **2.0s** between batches

## 🔧 Recommended Actions

### Option 1: Wait and Retry (Best)
Yahoo Finance rate limits reset after time. **Wait 15-30 minutes**, then run:

```bash
python main.py
```

### Option 2: Use Test Script First
Test with a small subset to verify it works:

```bash
python test_download.py
```

This downloads only 20 stocks to verify your connection works.

### Option 3: Use Very Conservative Settings
If still getting rate limited:

```bash
# Very safe settings (slower but reliable)
python main.py --batch-size 5 --delay 3.0
```

### Option 4: Try During Off-Peak Hours
Yahoo Finance may have less traffic during:
- Early morning (6-8 AM your timezone)
- Late evening (10 PM - midnight)
- Weekends

## 📊 Expected Download Times

| Batch Size | Delay | Time for ~500 stocks | Reliability |
|------------|-------|---------------------|-------------|
| 5 | 3.0s | 12-15 minutes | ⭐⭐⭐⭐⭐ Very High |
| 10 | 2.0s | 8-10 minutes | ⭐⭐⭐⭐ High |
| 20 | 1.5s | 5-7 minutes | ⭐⭐⭐ Medium |
| 50 | 1.0s | 3-5 minutes | ⭐ Low (may fail) |

## 🎯 Why This Happens

Yahoo Finance's free API has limitations:
1. **Request quotas**: Maximum requests per minute/hour
2. **IP-based throttling**: Tracks requests from your IP address
3. **Concurrent connections**: Limits simultaneous downloads
4. **Time-based cooldowns**: Enforces waiting periods

## 💡 Best Practice Workflow

```bash
# Step 1: Try to download with default settings
python main.py

# If rate limited, wait 30 minutes

# Step 2: Try again with conservative settings  
python main.py --batch-size 5 --delay 3.0

# Once downloaded, data is cached - subsequent runs are instant!
python main.py  # Uses cache (10-30 seconds)
```

## 🚀 The Caching Advantage

**Good news**: Once data is downloaded and cached, you won't need to download again!

```
First time:     5-15 minutes (depending on batch settings)
Second time:    10-30 seconds (uses cache) ✨
Third time:     10-30 seconds (uses cache) ✨
...and so on!
```

Cache is valid for 24 hours and updates automatically when needed.

## 🔍 Troubleshooting

### Still Getting Rate Limited?

1. **Check your recent activity**
   - Have you made many requests recently?
   - Clear cache and wait longer

2. **Try different batch sizes**
   ```bash
   # Start very small
   python main.py --batch-size 3 --delay 5.0
   ```

3. **Use a VPN or different network**
   - Rate limits may be IP-based
   - Try from a different location

4. **Check Yahoo Finance status**
   - Service may be experiencing issues
   - Check https://finance.yahoo.com/

### Alternative: Use Fewer Stocks

For testing, analyze a subset:
```bash
python test_download.py  # Only 20 stocks
```

Or modify `main.py` to download fewer tickers temporarily.

## 📝 Summary

**The fix I applied:**
- ✅ Changed defaults from batch_size=50 to 10
- ✅ Changed delay from 1.0s to 2.0s  
- ✅ Added retry logic with exponential backoff
- ✅ Created test script for small downloads
- ✅ Better error messages

**What you should do:**
1. Wait 15-30 minutes (let rate limit reset)
2. Run: `python main.py` (now uses conservative defaults)
3. If still fails: `python main.py --batch-size 5 --delay 3.0`
4. Once cached, enjoy fast subsequent runs!

## 🆘 If Nothing Works

If you continue to experience rate limiting issues even with conservative settings:

1. **Consider using paid API** (like Alpha Vantage, Polygon.io)
2. **Download incrementally** (50 stocks at a time over several days)
3. **Use pre-downloaded datasets** (Kaggle, Quandl)
4. **Contact Yahoo Finance** to check if your IP is blocked

The caching system ensures you only need to download once successfully!

