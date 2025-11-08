# Gold Predictor - Quick Start Guide

Get predictions in 3 simple steps!

## Step 1: Get API Keys (5 minutes)

### Polygon.io (Required)
1. Sign up at https://polygon.io (free tier)
2. Copy your API key

### FRED (Optional but Recommended)
1. Sign up at https://fred.stlouisfed.org
2. Go to https://fred.stlouisfed.org/docs/api/api_key.html
3. Request free API key

### Configure
Create/edit `.env` file in project root:
```bash
cd /path/to/vibestock
echo "POLYGON_API_KEY=your_polygon_key_here" >> .env
echo "FRED_API_KEY=your_fred_key_here" >> .env
```

## Step 2: Install Dependencies (1 minute)

```bash
# From project root
conda activate vibestock
pip install fredapi scikit-learn
```

## Step 3: Train & Predict (5 minutes first time, 30 seconds after)

```bash
cd gold_predictor

# Train model (first time only)
python predictor.py --train

# Make predictions
python predictor.py --predict
```

## Example Output

```
PREDICTIONS
================================================================================

📈 1 Week     (SHORT-term)
   Prediction:  UP
   Probability: 67.3% UP  /  32.7% DOWN
   Confidence:  67.3% (MEDIUM)

📈 1 Month    (MID-term)
   Prediction:  UP
   Probability: 72.1% UP  /  27.9% DOWN
   Confidence:  72.1% (HIGH)

📉 3 Months   (LONG-term)
   Prediction:  DOWN
   Probability: 42.8% UP  /  57.2% DOWN
   Confidence:  57.2% (LOW)

KEY ECONOMIC INDICATORS (Current Values)
================================================================================

  Real Interest Rate         :    -1.20
  Fed Funds Rate            :     5.25
  CPI Inflation Rate        :     3.50
  Inflation Expectations    :     2.45
  USD 30-day Return         :     2.15
  VIX (Fear Index)          :    18.50
  Gold 30-day Volatility    :    12.30
```

## Understanding Results

### High Confidence (>70%)
✅ Strong signal - trust the prediction more

### Medium Confidence (60-70%)
⚠️ Moderate signal - consider other factors

### Low Confidence (<60%)
❓ Weak signal - market is uncertain

### Time Horizons
- **Short (1 week)**: More influenced by technical factors and sentiment
- **Mid (1 month)**: Balance of technical and fundamental
- **Long (3 months)**: Driven by economic fundamentals

## Next Steps

### Retrain Weekly
```bash
python predictor.py --retrain
```

### Evaluate Performance
```bash
python predictor.py --evaluate
```

### Compare Models
```bash
# Train random forest
python predictor.py --train --model-type random_forest

# Compare predictions
python predictor.py --predict --model-path models/gold_predictor.pkl
python predictor.py --predict --model-path models/gold_predictor_rf.pkl
```

## Tips

1. **Check all horizons** - If all three agree, stronger signal
2. **Look at context** - Understand WHY the model predicts what it does
3. **Monitor changes** - Retrain weekly to see evolving predictions
4. **Use confidence** - Higher confidence = more reliable

## Troubleshooting

### "Model not found"
Run training first: `python predictor.py --train`

### "FRED API key not found"
Still works! Just uses fewer features. Get key at:
https://fred.stlouisfed.org/docs/api/api_key.html

### Low accuracy
- Try random forest: `--model-type random_forest`
- Train with more data: `--years 10`
- Add FRED API key for more features

## Full Documentation

- **[README.md](README.md)** - Complete documentation
- **[INTERPRETATION_GUIDE.md](docs/INTERPRETATION_GUIDE.md)** - Deep dive into economics

---

**Remember**: This is a tool to inform decisions, not financial advice! 📊✨

