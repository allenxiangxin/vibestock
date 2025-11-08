# vibestock

Two powerful stock analysis tools using professional-grade Polygon.io data.

## Projects

### 📊 S&P 500 Analyzer
Analyze return distributions across S&P 500 stocks.

```bash
cd sp500_analyzer
python main.py
```

[Learn more →](sp500_analyzer/)

### 🥇 Gold Tracker
Monitor gold prices with MACD indicator and get desktop notifications.

```bash
cd gold_tracker
python gold_tracker.py --daemon
```

[Learn more →](gold_tracker/)

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
├── docs/               # Shared documentation
├── archive/            # Old files (reference only)
└── requirements.txt    # Dependencies
```

## Documentation

- **[S&P 500 Analyzer Guide](sp500_analyzer/README.md)** - Complete analyzer documentation
- **[Gold Tracker Guide](gold_tracker/README.md)** - Complete tracker documentation
- **[Polygon.io Setup](docs/POLYGON_SETUP.md)** - API configuration guide
- **[Reorganization Summary](REORGANIZATION_SUMMARY.md)** - What changed and where files moved

## Requirements

- Python 3.8+
- Polygon.io API key (free tier available)
- See [requirements.txt](requirements.txt) for packages

## License

MIT License
