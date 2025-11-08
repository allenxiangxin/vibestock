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

### Option A: Single Model (Recommended for Beginners)

```bash
cd gold_predictor

# Train model with optimal 8 years of data (first time only)
# Uses Random Forest by default for 97% accuracy
python predictor.py --train --years 8

# Make predictions (uses saved Random Forest model)
python predictor.py --predict
```

**Note:** Random Forest is now the default (recommended). The model type is saved when you train, so future predictions automatically use the same model.

### Option B: Compare ALL Models (For Advanced Users)

Train BOTH Logistic Regression and Random Forest to compare:

```bash
# Train BOTH models (takes 6-10 minutes)
python predictor.py --train --all-models --years 8

# Get predictions from BOTH models with side-by-side comparison
python predictor.py --predict --all-models
```

This shows you how both models compare and gives you more confidence when they agree!

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

### Compare ALL Models (Logistic vs Random Forest)

Train and compare BOTH model types to see which performs better:

```bash
# Train BOTH models at once (takes 6-10 minutes)
python predictor.py --train --all-models --years 8

# Get predictions from BOTH models with comparison
python predictor.py --predict --all-models

# Evaluate BOTH models with accuracy comparison
python predictor.py --evaluate --all-models
```

**Example comparison output:**
```
📊 MODEL COMPARISON
────────────────────────────────────────────────────────────────
⏰ 1 Week Predictions:
────────────────────────────────────────────────────────────────
  logistic       : ⬆️ UP    (Confidence: 67.3%)
  random_forest  : ⬆️ UP    (Confidence: 92.1%)
```

**When to use:**
- ✅ **Both agree** → High confidence signal!
- ⚠️ **Models disagree** → Market is uncertain, be cautious
- 📊 **For trading** → Use Random Forest (97% accuracy)
- 📈 **For analysis** → Use Logistic (see feature importance)

See **[ALL_MODELS_GUIDE.md](ALL_MODELS_GUIDE.md)** for complete guide.

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
- Adjust training period: `--years 5` (recent) or `--years 10` (more data)
- Add FRED API key for more features (recommended: 8 years optimal)

## Full Documentation

- **[README.md](README.md)** - Complete documentation
- **[ALL_MODELS_GUIDE.md](ALL_MODELS_GUIDE.md)** - Compare all models (Logistic vs Random Forest)
- **[INTERPRETATION_GUIDE.md](docs/INTERPRETATION_GUIDE.md)** - Deep dive into economics
- **[USAGE.md](USAGE.md)** - Daily usage guide

---

**Remember**: This is a tool to inform decisions, not financial advice! 📊✨

