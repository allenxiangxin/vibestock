"""
Gold Price Predictor

Statistical model for predicting gold price direction based on economic fundamentals.
"""

__version__ = "1.0.0"

from .data_fetcher import fetch_all_data, merge_all_data, download_gold_data
from .feature_engineering import engineer_all_features, select_features_for_model
from .model import GoldPricePredictor

__all__ = [
    'fetch_all_data',
    'merge_all_data', 
    'download_gold_data',
    'engineer_all_features',
    'select_features_for_model',
    'GoldPricePredictor',
]

