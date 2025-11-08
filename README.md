# vibestock

Three powerful tools for stock and commodities analysis using professional-grade data.

## Projects

### 📊 S&P 500 Analyzer
Analyze return distributions across S&P 500 stocks with comprehensive statistics and visualizations.

```bash
cd sp500_analyzer
python main.py
```

[Learn more →](sp500_analyzer/)

### 🥇 Gold Tracker
Monitor gold prices with MACD technical indicator and get desktop notifications for buy signals.

```bash
cd gold_tracker
python gold_tracker.py --daemon
```

[Learn more →](gold_tracker/)

### 🔮 Gold Predictor
Predict gold price direction using statistical models based on economic fundamentals (real interest rates, USD strength, inflation, Fed policy).

```bash
cd gold_predictor
python predictor.py --train
python predictor.py --predict
```

[Learn more →](gold_predictor/)

## Quick Setup

### 1. Install Dependencies

```bash
conda create -n vibestock python=3.9 -y
conda activate vibestock
pip install -r requirements.txt
```

### 2. Configure API Key

Get a free API key from [polygon.io](https://polygon.io), then:

```bash
# Create .env file
echo "POLYGON_API_KEY=your_api_key_here" > .env
```

[Detailed setup guide →](docs/POLYGON_SETUP.md)

## Project Structure

```
vibestock/
├── sp500_analyzer/     # S&P 500 return analysis
├── gold_tracker/       # Gold price MACD tracker  
├── gold_predictor/     # Gold price ML predictor
├── docs/               # Shared documentation
├── archive/            # Old files (reference only)
└── requirements.txt    # Dependencies
```

## Documentation

- **[S&P 500 Analyzer Guide](sp500_analyzer/README.md)** - Statistical return analysis
- **[Gold Tracker Guide](gold_tracker/README.md)** - MACD technical analysis  
- **[Gold Predictor Guide](gold_predictor/README.md)** - ML-based fundamental analysis
- **[Polygon.io Setup](docs/POLYGON_SETUP.md)** - API configuration guide
- **[Reorganization Summary](REORGANIZATION_SUMMARY.md)** - What changed and where files moved

## Requirements

- Python 3.8+
- Polygon.io API key (free tier available)
- FRED API key (optional, for gold predictor - improves accuracy)
- See [requirements.txt](requirements.txt) for packages

## License

MIT License
