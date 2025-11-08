#!/usr/bin/env python3
"""
Gold Price Predictor

Statistical model for predicting gold price direction based on economic fundamentals.

Usage:
    python predictor.py --train              # Train new model
    python predictor.py --predict            # Make prediction with existing model
    python predictor.py --retrain            # Retrain with latest data
    python predictor.py --evaluate           # Evaluate model performance
"""

import argparse
import sys
import os
from datetime import datetime
import pandas as pd

# Import our modules
from data_fetcher import fetch_all_data, merge_all_data
from feature_engineering import engineer_all_features, select_features_for_model
from model import GoldPricePredictor, evaluate_model


def train_all_models(lookback_years: int = 8):
    """
    Train ALL model types (logistic + random_forest) for comparison.
    
    Args:
        lookback_years: Years of historical data to use
    """
    print(f"\n{'='*80}")
    print("GOLD PRICE PREDICTOR - TRAINING ALL MODELS")
    print(f"{'='*80}\n")
    print(f"Training both Logistic Regression and Random Forest...")
    print(f"Lookback: {lookback_years} years\n")
    
    models_trained = {}
    
    for model_type in ['logistic', 'random_forest']:
        print(f"\n{'#'*80}")
        print(f"# TRAINING: {model_type.upper()}")
        print(f"{'#'*80}\n")
        
        save_path = f'models/gold_predictor_{model_type}.pkl'
        predictor = train_model(lookback_years, model_type, save_path)
        
        if predictor:
            models_trained[model_type] = save_path
    
    print(f"\n{'='*80}")
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"\nModels saved:")
    for model_type, path in models_trained.items():
        print(f"  • {model_type:15s}: {path}")
    print()
    
    return models_trained


def train_model(lookback_years: int = 8, model_type: str = 'random_forest', 
                save_path: str = 'models/gold_predictor.pkl'):
    """
    Train a new model from scratch.
    
    Args:
        lookback_years: Years of historical data to use
        model_type: 'logistic' or 'random_forest'
        save_path: Where to save the trained model
    """
    print(f"\n{'='*80}")
    print("GOLD PRICE PREDICTOR - TRAINING")
    print(f"{'='*80}\n")
    print(f"Model type: {model_type}")
    print(f"Lookback: {lookback_years} years")
    print(f"Save path: {save_path}\n")
    
    # Step 1: Fetch data
    print("STEP 1: Fetching data...")
    data = fetch_all_data(lookback_years=lookback_years)
    
    # Step 2: Merge data
    print("\nSTEP 2: Merging data sources...")
    df_merged = merge_all_data(data)
    
    # Save merged data
    os.makedirs('data', exist_ok=True)
    df_merged.to_csv('data/gold_predictor_data.csv', index=False)
    
    # Step 3: Engineer features
    print("\nSTEP 3: Engineering features...")
    df_features = engineer_all_features(df_merged)
    
    # Save features
    df_features.to_csv('data/gold_features.csv', index=False)
    
    # Step 4: Prepare training data
    print("\nSTEP 4: Preparing training data...")
    horizons = ['short', 'mid', 'long']
    y_dict = {}
    
    for horizon in horizons:
        target_col = f'target_{horizon}'
        if target_col in df_features.columns:
            X, y = select_features_for_model(df_features, target_col)
            y_dict[horizon] = y
    
    print(f"   Training samples: {len(X)}")
    print(f"   Features: {len(X.columns)}")
    print(f"   Targets: {list(y_dict.keys())}")
    
    # Step 5: Train model
    print("\nSTEP 5: Training model...")
    predictor = GoldPricePredictor(model_type=model_type)
    results = predictor.train(X, y_dict, test_size=0.2)
    
    # Check if any models were trained
    if not predictor.models:
        print("\n" + "="*80)
        print("⚠️  WARNING: No models were trained!")
        print("="*80)
        print("\nThis usually happens when gold prices moved in only one direction.")
        print("Try one of these solutions:")
        print("  1. Use more years of data: --years 10")
        print("  2. Wait for more market data to accumulate")
        print("  3. Check if FRED API key is configured for more features")
        print("\nExample: python predictor.py --train --years 10")
        return None
    
    # Step 6: Save model
    print("\nSTEP 6: Saving model...")
    predictor.save_model(save_path)
    
    # Step 7: Show feature importance
    print("\n" + "="*80)
    print("TOP 10 MOST IMPORTANT FEATURES (by horizon)")
    print("="*80 + "\n")
    
    for horizon in horizons:
        if horizon in predictor.models:
            print(f"{horizon.upper()}-term:")
            top_features = predictor.get_top_features(horizon, n=10)
            for idx, row in top_features.iterrows():
                print(f"  {idx+1:2d}. {row['Feature']:35s}  {row['Importance']:.4f}")
            print()
    
    print("="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80 + "\n")
    
    return predictor


def predict_all_models(show_details: bool = True):
    """
    Make predictions using ALL trained models for comparison.
    
    Args:
        show_details: Whether to show detailed feature values
    """
    print(f"\n{'='*80}")
    print("GOLD PRICE PREDICTOR - COMPARING ALL MODELS")
    print(f"{'='*80}\n")
    
    model_paths = {
        'logistic': 'models/gold_predictor_logistic.pkl',
        'random_forest': 'models/gold_predictor_random_forest.pkl'
    }
    
    # Fetch data ONCE and reuse for both models (avoid API rate limits)
    print("Fetching latest data...")
    try:
        data = fetch_all_data(lookback_years=2)
        df_merged = merge_all_data(data)
        df_features = engineer_all_features(df_merged)
        
        if len(df_features) == 0:
            print("\n❌ ERROR: No data available after feature engineering!")
            return None
            
        print(f"✅ Data fetched successfully ({len(df_features)} observations)\n")
    except Exception as e:
        print(f"\n❌ ERROR fetching data: {e}")
        return None
    
    predictions_by_model = {}
    
    for model_type, model_path in model_paths.items():
        if os.path.exists(model_path):
            print(f"\n{'─'*80}")
            print(f"📊 {model_type.upper()} MODEL")
            print(f"{'─'*80}")
            result = make_prediction_with_data(model_path, df_features, show_details=False)
            if result:
                predictions_by_model[model_type] = result
        else:
            print(f"\n⚠️  {model_type.upper()} model not found at: {model_path}")
    
    if len(predictions_by_model) > 1:
        print(f"\n{'='*80}")
        print("📊 MODEL COMPARISON")
        print(f"{'='*80}\n")
        
        horizons = ['short', 'mid', 'long']
        horizon_names = {'short': '1 Week', 'mid': '1 Month', 'long': '3 Months'}
        
        for horizon in horizons:
            print(f"{'─'*80}")
            print(f"⏰ {horizon_names[horizon]} Predictions:")
            print(f"{'─'*80}")
            
            for model_type in predictions_by_model.keys():
                if horizon in predictions_by_model[model_type]:
                    pred = predictions_by_model[model_type][horizon]
                    direction = pred['prediction']
                    conf = pred['confidence']
                    icon = "⬆️" if direction == 'UP' else "⬇️"
                    
                    print(f"  {model_type:15s}: {icon} {direction:4s}  (Confidence: {conf:.1%})")
            print()
    
    return predictions_by_model


def make_prediction_with_data(model_path: str, df_features: pd.DataFrame, show_details: bool = True):
    """
    Make a prediction using a trained model with pre-fetched data.
    
    Args:
        model_path: Path to saved model
        df_features: Pre-fetched and engineered features DataFrame
        show_details: Whether to show detailed feature values
        
    Returns:
        Dictionary with predictions for each horizon
    """
    # Load model
    try:
        predictor = GoldPricePredictor.load_model(model_path)
    except FileNotFoundError:
        if show_details:
            print(f"\n❌ ERROR: Model file not found: {model_path}")
        return None
    
    # Check if all required features are present
    missing_features = [f for f in predictor.feature_names if f not in df_features.columns]
    if missing_features:
        print(f"\n❌ ERROR: Required features missing from data: {missing_features}")
        print(f"\nAvailable columns: {list(df_features.columns)}")
        print("\nThis usually means:")
        print("  - USD Index (UUP) data failed to download")
        print("  - FRED economic data failed to download")
        print("\nTry running again or check your API keys in .env file")
        return None
    
    # Get the most recent complete data point
    latest_data = df_features[predictor.feature_names].iloc[[-1]]
    latest_date = df_features['date'].iloc[-1]
    latest_price = df_features['close'].iloc[-1]
    
    if show_details:
        print(f"\nData as of: {latest_date.strftime('%Y-%m-%d')}")
        print(f"Current gold price (GLD): ${latest_price:.2f}")
    
    # Make predictions
    if show_details:
        print("\n" + "="*80)
        print("PREDICTIONS")
        print("="*80 + "\n")
    
    horizons = ['short', 'mid', 'long']
    horizon_names = {'short': '1 Week', 'mid': '1 Month', 'long': '3 Months'}
    
    predictions = {}
    
    for horizon in horizons:
        if horizon in predictor.models:
            pred = predictor.predict(latest_data, horizon)
            predictions[horizon] = pred
            
            if show_details:
                direction = pred['prediction']
                prob_up = pred['probability_up']
                prob_down = pred['probability_down']
                confidence = pred['confidence']
                
                # Determine confidence level
                if confidence >= 0.70:
                    conf_level = "HIGH"
                elif confidence >= 0.60:
                    conf_level = "MEDIUM"
                else:
                    conf_level = "LOW"
                
                # Format output
                arrow = "📈" if direction == "UP" else "📉"
                
                print(f"{arrow} {horizon_names[horizon]:10s} ({horizon.upper()}-term)")
                print(f"   Prediction:  {direction}")
                print(f"   Probability: {prob_up:.1%} UP  /  {prob_down:.1%} DOWN")
                print(f"   Confidence:  {confidence:.1%} ({conf_level})")
                print()
    
    # Show key features if requested
    if show_details:
        print("="*80)
        print("KEY ECONOMIC INDICATORS (Current Values)")
        print("="*80 + "\n")
        
        key_features = {
            'real_interest_rate': 'Real Interest Rate',
            'fed_funds_rate': 'Fed Funds Rate',
            'inflation_rate': 'CPI Inflation Rate',
            'inflation_expectations': 'Inflation Expectations',
            'usd_close_return_30d': 'USD 30-day Return',
            'vix': 'VIX (Fear Index)',
            'close_vol_30d': 'Gold 30-day Volatility'
        }
        
        for feat, name in key_features.items():
            if feat in latest_data.columns:
                value = latest_data[feat].iloc[0]
                print(f"  {name:30s}: {value:8.2f}")
    
        print("\n" + "="*80)
        print("INTERPRETATION GUIDE")
        print("="*80)
        print("""
Gold Typically Rises When:
  ↓ Real interest rates fall (negative is bullish)
  ↓ USD weakens (negative return is bullish for gold)
  ↑ Inflation expectations increase
  ↑ VIX increases (fear/uncertainty)
  ↑ Fed keeps rates low or cuts

Gold Typically Falls When:
  ↑ Real interest rates rise (positive is bearish)
  ↑ USD strengthens (positive return is bearish for gold)
  ↓ Inflation expectations decrease
  ↓ VIX decreases (calm markets)
  ↑ Fed raises rates aggressively
""")
        print("="*80 + "\n")
    
    return predictions


def make_prediction(model_path: str = 'models/gold_predictor.pkl', 
                   show_details: bool = True):
    """
    Make a prediction using a trained model.
    
    Args:
        model_path: Path to saved model
        show_details: Whether to show detailed feature values
    """
    if show_details:
        print(f"\n{'='*80}")
        print("GOLD PRICE PREDICTOR - PREDICTION")
        print(f"{'='*80}\n")
        print(f"Loading model from: {model_path}")
    
    # Fetch latest data (need 2 years for 200-day MA and other features)
    if show_details:
        print("\nFetching latest data...")
    
    try:
        data = fetch_all_data(lookback_years=2)  # Need enough history for feature calculation
        df_merged = merge_all_data(data)
        
        if show_details:
            print("\nEngineering features...")
        df_features = engineer_all_features(df_merged)
        
        # Check if we have any data after feature engineering
        if len(df_features) == 0:
            print("\n❌ ERROR: No data available after feature engineering!")
            print("This usually means:")
            print("  - Not enough historical data to calculate features")
            print("  - Try fetching more years of data")
            return None
    except Exception as e:
        print(f"\n❌ ERROR fetching/processing data: {e}")
        return None
    
    # Use the helper function with the fetched data
    return make_prediction_with_data(model_path, df_features, show_details)


def evaluate_all_models():
    """
    Evaluate ALL trained models and compare their performance.
    """
    print(f"\n{'='*80}")
    print("GOLD PRICE PREDICTOR - EVALUATING ALL MODELS")
    print(f"{'='*80}\n")
    
    model_paths = {
        'Logistic Regression': 'models/gold_predictor_logistic.pkl',
        'Random Forest': 'models/gold_predictor_random_forest.pkl'
    }
    
    results_by_model = {}
    
    for model_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            print(f"\n{'#'*80}")
            print(f"# EVALUATING: {model_name.upper()}")
            print(f"{'#'*80}\n")
            result = evaluate_existing_model(model_path)
            if result:
                results_by_model[model_name] = result
        else:
            print(f"\n⚠️  {model_name} model not found at: {model_path}")
            print(f"    Train it with: python predictor.py --train --model-type {model_name.lower().replace(' ', '_')}\n")
    
    # Print comparison summary
    if len(results_by_model) > 1:
        print(f"\n{'='*80}")
        print("📊 MODEL ACCURACY COMPARISON")
        print(f"{'='*80}\n")
        
        horizons = {'short': '1 Week', 'mid': '1 Month', 'long': '3 Months'}
        
        print(f"{'Time Horizon':<20} {'Logistic Regression':<25} {'Random Forest':<25}")
        print(f"{'-'*70}")
        
        for horizon, horizon_name in horizons.items():
            line = f"{horizon_name:<20}"
            
            for model_name in ['Logistic Regression', 'Random Forest']:
                if model_name in results_by_model and horizon in results_by_model[model_name]:
                    accuracy = results_by_model[model_name][horizon]
                    line += f" {accuracy:>6.1%}                  "
                else:
                    line += f" {'N/A':>6}                  "
            
            print(line)
        
        print()
    
    return results_by_model


def evaluate_existing_model(model_path: str = 'models/gold_predictor.pkl'):
    """
    Evaluate a trained model on recent data.
    
    Args:
        model_path: Path to saved model
    """
    print(f"\n{'='*80}")
    print("GOLD PRICE PREDICTOR - EVALUATION")
    print(f"{'='*80}\n")
    
    # Load model
    try:
        predictor = GoldPricePredictor.load_model(model_path)
    except FileNotFoundError:
        print(f"\n❌ ERROR: Model file not found: {model_path}")
        return
    
    # Load feature data
    try:
        df = pd.read_csv('data/gold_features.csv')
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Loaded {len(df)} rows of feature data")
    except FileNotFoundError:
        print("❌ ERROR: No feature data found. Run training first.")
        return
    
    # Prepare test data (most recent 20%)
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:]
    
    print(f"   Evaluation period: {df_test['date'].min()} to {df_test['date'].max()}")
    print(f"   Test samples: {len(df_test)}\n")
    
    # Prepare targets
    horizons = ['short', 'mid', 'long']
    X_test = df_test[predictor.feature_names]
    y_test_dict = {}
    
    for horizon in horizons:
        target_col = f'target_{horizon}'
        if target_col in df_test.columns:
            y_test_dict[horizon] = df_test[target_col]
    
    # Evaluate
    evaluate_model(predictor, X_test, y_test_dict)
    
    # Calculate and return accuracy for each horizon
    accuracies = {}
    for horizon in predictor.models.keys():
        y_test = y_test_dict[horizon]
        model = predictor.models[horizon]
        scaler = predictor.scalers[horizon]
        
        X_test_aligned = X_test[predictor.feature_names]
        X_test_scaled = scaler.transform(X_test_aligned)
        y_pred = model.predict(X_test_scaled)
        
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[horizon] = accuracy
    
    return accuracies


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Gold Price Predictor - Statistical model based on economic fundamentals"
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train a new model from scratch'
    )
    
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Make predictions with existing model'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate existing model performance'
    )
    
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Retrain model with latest data'
    )
    
    parser.add_argument(
        '--all-models',
        action='store_true',
        help='Train/predict/evaluate ALL model types (logistic + random_forest) for comparison'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='random_forest',
        choices=['logistic', 'random_forest'],
        help='Type of model to train (default: random_forest, recommended for best accuracy)'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        default=8,
        help='Years of historical data for training (default: 8, optimal range: 7-10 years)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/gold_predictor.pkl',
        help='Path to model file (default: models/gold_predictor.pkl)'
    )
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any([args.train, args.predict, args.evaluate, args.retrain]):
        parser.print_help()
        print("\n" + "="*80)
        print("QUICK START:")
        print("="*80)
        print("  1. Train a model:        python predictor.py --train")
        print("  2. Make predictions:     python predictor.py --predict")
        print("  3. Evaluate performance: python predictor.py --evaluate")
        print("  4. Retrain with new data: python predictor.py --retrain")
        print("="*80 + "\n")
        return
    
    try:
        # Handle --all-models flag
        if args.all_models:
            if args.train or args.retrain:
                train_all_models(lookback_years=args.years)
            elif args.predict:
                predict_all_models(show_details=True)
            elif args.evaluate:
                evaluate_all_models()
            else:
                print("\n⚠️  --all-models requires --train, --predict, or --evaluate\n")
        else:
            # Single model mode
            if args.train or args.retrain:
                train_model(
                    lookback_years=args.years,
                    model_type=args.model_type,
                    save_path=args.model_path
                )
            
            if args.predict:
                make_prediction(model_path=args.model_path)
            
            if args.evaluate:
                evaluate_existing_model(model_path=args.model_path)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

