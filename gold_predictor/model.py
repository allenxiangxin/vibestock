"""
Gold Predictor - Statistical Model Module

Trains and evaluates models for gold price direction prediction.
Uses logistic regression for interpretability, random forest for accuracy,
and gradient boosting (GBDT) for maximum performance.
Includes SMOTE for handling class imbalance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# Try to import SMOTE for handling class imbalance
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️  WARNING: imbalanced-learn not installed. Class balancing with SMOTE unavailable.")
    print("   Install with: pip install imbalanced-learn")


class GoldPricePredictor:
    """
    Statistical model for predicting gold price direction.
    """
    
    def __init__(self, model_type: str = 'logistic'):
        """
        Initialize the predictor.
        
        Args:
            model_type: 'logistic', 'random_forest', or 'gbdt'
        """
        self.model_type = model_type
        self.models = {}  # One model per time horizon
        self.scalers = {}  # One scaler per time horizon
        self.feature_names = []
        self.feature_importance = {}
        
    def train(self, X: pd.DataFrame, y_dict: Dict[str, pd.Series], 
              test_size: float = 0.2, random_state: int = 42, use_smote: bool = True):
        """
        Train models for each time horizon.
        
        Args:
            X: Feature DataFrame
            y_dict: Dictionary mapping horizon names to target Series
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.feature_names = X.columns.tolist()
        results = {}
        
        print(f"\n{'='*80}")
        print(f"TRAINING {self.model_type.upper()} MODELS")
        print(f"{'='*80}\n")
        
        for horizon, y in y_dict.items():
            print(f"📊 Training {horizon}-term model...")
            
            # Check class distribution
            unique_classes, class_counts = np.unique(y, return_counts=True)
            print(f"   Class distribution: DOWN={class_counts[0] if len(class_counts) > 0 else 0}, "
                  f"UP={class_counts[1] if len(class_counts) > 1 else 0}")
            
            # Check if we have both classes
            if len(unique_classes) < 2:
                print(f"   ⚠️  WARNING: Only one class present in data. Skipping {horizon}-term model.")
                print(f"   This means gold prices only moved in one direction during the training period.")
                continue
            
            # Use stratified split to ensure both classes in train and test
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, 
                    stratify=y, shuffle=True  # stratify ensures both classes in splits
                )
            except ValueError:
                # If stratification fails, use time-based split
                print(f"   ⚠️  Stratified split failed, using time-based split")
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Check test set has both classes
            test_classes = np.unique(y_test)
            if len(test_classes) < 2:
                print(f"   ⚠️  WARNING: Test set only has one class. Results may be unreliable.")
                print(f"   Consider using more training data or different time period.")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if self.model_type == 'logistic':
                model = LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                    class_weight='balanced'
                )
            elif self.model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=random_state,
                    class_weight='balanced'
                )
            elif self.model_type == 'gbdt':
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    subsample=0.8,
                    random_state=random_state
                )
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")
            
            # Apply SMOTE if available and requested (only for training data)
            X_train_final = X_train_scaled
            y_train_final = y_train
            
            if use_smote and SMOTE_AVAILABLE and len(unique_classes) == 2:
                # Count samples in each class
                class_counts = y_train.value_counts()
                min_samples = class_counts.min()
                
                # Only apply SMOTE if we have enough samples in minority class
                if min_samples >= 6:  # SMOTE needs at least 6 samples for k_neighbors=5
                    try:
                        smote = SMOTE(random_state=random_state, k_neighbors=min(5, min_samples-1))
                        X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
                        print(f"   🔄 SMOTE applied: {len(y_train)} → {len(y_train_final)} samples")
                        print(f"      Class balance: DOWN={sum(y_train_final==0)}, UP={sum(y_train_final==1)}")
                    except Exception as e:
                        print(f"   ⚠️  SMOTE failed: {e}, using original data")
                        X_train_final = X_train_scaled
                        y_train_final = y_train
                else:
                    print(f"   ⚠️  SMOTE skipped: minority class has only {min_samples} samples (need 6+)")
            
            model.fit(X_train_final, y_train_final)
            
            # Evaluate
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
            
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            
            # Cross-validation with stratification
            try:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf)
            except ValueError as e:
                print(f"   ⚠️  Cross-validation warning: {e}")
                cv_scores = np.array([train_acc])  # Fallback to train accuracy
            
            # Store results
            results[horizon] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred_test,
                'probabilities': y_prob_test,
                'actual': y_test
            }
            
            # Store model and scaler
            self.models[horizon] = model
            self.scalers[horizon] = scaler
            
            # Get feature importance
            if self.model_type == 'logistic':
                importance = np.abs(model.coef_[0])
            else:  # random_forest or gbdt
                importance = model.feature_importances_
            
            self.feature_importance[horizon] = dict(zip(self.feature_names, importance))
            
            # Print results
            print(f"   Train Accuracy: {train_acc:.3f}")
            print(f"   Test Accuracy:  {test_acc:.3f}")
            print(f"   CV Accuracy:    {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
            print(f"\n   Classification Report:")
            print("   " + "-"*60)
            
            # Get labels present in test set
            labels_present = np.unique(y_test)
            target_names_used = ['Down' if i == 0 else 'Up' for i in labels_present]
            
            try:
                report = classification_report(y_test, y_pred_test, 
                                              labels=labels_present,
                                              target_names=target_names_used, 
                                              zero_division=0)
                for line in report.split('\n'):
                    if line:
                        print(f"   {line}")
            except Exception as e:
                print(f"   ⚠️  Could not generate full report: {e}")
                print(f"   Accuracy: {test_acc:.3f}")
            print()
        
        print(f"{'='*80}\n")
        
        return results
    
    def predict(self, X: pd.DataFrame, horizon: str = 'mid') -> Dict:
        """
        Make predictions for new data.
        
        Args:
            X: Feature DataFrame
            horizon: Time horizon ('short', 'mid', 'long')
            
        Returns:
            Dictionary with prediction, probability, and confidence
        """
        if horizon not in self.models:
            raise ValueError(f"No model trained for horizon: {horizon}")
        
        model = self.models[horizon]
        scaler = self.scalers[horizon]
        
        # Ensure X has the same features in the same order
        X_aligned = X[self.feature_names].copy()
        
        # Scale
        X_scaled = scaler.transform(X_aligned)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        return {
            'prediction': 'UP' if prediction == 1 else 'DOWN',
            'probability_up': probabilities[1],
            'probability_down': probabilities[0],
            'confidence': max(probabilities)
        }
    
    def predict_all_horizons(self, X: pd.DataFrame) -> Dict:
        """
        Make predictions for all time horizons.
        
        Args:
            X: Feature DataFrame (typically the most recent data point)
            
        Returns:
            Dictionary with predictions for each horizon
        """
        predictions = {}
        
        for horizon in self.models.keys():
            predictions[horizon] = self.predict(X, horizon)
        
        return predictions
    
    def get_top_features(self, horizon: str, n: int = 10) -> pd.DataFrame:
        """
        Get top N most important features for a given horizon.
        
        Args:
            horizon: Time horizon
            n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if horizon not in self.feature_importance:
            raise ValueError(f"No model trained for horizon: {horizon}")
        
        importance_dict = self.feature_importance[horizon]
        
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), 
                               key=lambda x: x[1], reverse=True)[:n]
        
        return pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        model_data = {
            'model_type': self.model_type,
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls(model_type=model_data['model_type'])
        predictor.models = model_data['models']
        predictor.scalers = model_data['scalers']
        predictor.feature_names = model_data['feature_names']
        predictor.feature_importance = model_data['feature_importance']
        
        print(f"✅ Model loaded from: {filepath}")
        
        return predictor


def evaluate_model(predictor: GoldPricePredictor, X_test: pd.DataFrame, 
                   y_test_dict: Dict[str, pd.Series]):
    """
    Evaluate a trained model and print detailed metrics.
    
    Args:
        predictor: Trained GoldPricePredictor
        X_test: Test features
        y_test_dict: Dictionary of test targets for each horizon
    """
    print(f"\n{'='*80}")
    print("DETAILED MODEL EVALUATION")
    print(f"{'='*80}\n")
    
    for horizon in predictor.models.keys():
        print(f"📊 {horizon.upper()}-TERM MODEL")
        print("-" * 80)
        
        y_test = y_test_dict[horizon]
        
        # Get predictions
        scaler = predictor.scalers[horizon]
        model = predictor.models[horizon]
        
        X_test_aligned = X_test[predictor.feature_names]
        X_test_scaled = scaler.transform(X_test_aligned)
        
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        
        # Check if we have both classes
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        if len(unique_labels) == 2 and cm.shape == (2, 2):
            # Full 2x2 matrix
            print(f"                  Predicted")
            print(f"                  Down    Up")
            print(f"Actual    Down    {cm[0,0]:4d}  {cm[0,1]:4d}")
            print(f"          Up      {cm[1,0]:4d}  {cm[1,1]:4d}")
        elif len(unique_labels) == 1:
            # Only one class present
            label_name = "Down" if unique_labels[0] == 0 else "Up"
            print(f"⚠️  WARNING: Only one class present in test set ({label_name})")
            print(f"All predictions: {label_name}  Count: {cm[0,0]:4d}")
        else:
            # Fallback
            print(f"Shape: {cm.shape}")
            print(cm)
        
        # Top features
        print("\nTop 10 Most Important Features:")
        top_features = predictor.get_top_features(horizon, n=10)
        for idx, row in top_features.iterrows():
            print(f"  {idx+1:2d}. {row['Feature']:30s}  {row['Importance']:.4f}")
        
        print("\n")


if __name__ == "__main__":
    """Test model training."""
    print("Testing Gold Price Prediction Model\n")
    
    # Load features
    try:
        df = pd.read_csv('data/gold_features.csv')
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Loaded {len(df)} rows of feature data\n")
    except FileNotFoundError:
        print("ERROR: Run data_fetcher.py and feature_engineering.py first")
        import sys
        sys.exit(1)
    
    # Import feature selection
    from feature_engineering import select_features_for_model
    
    # Prepare data for each horizon
    horizons = ['short', 'mid', 'long']
    y_dict = {}
    
    for horizon in horizons:
        target_col = f'target_{horizon}'
        if target_col in df.columns:
            X, y = select_features_for_model(df, target_col)
            y_dict[horizon] = y
    
    # Train model
    predictor = GoldPricePredictor(model_type='logistic')
    results = predictor.train(X, y_dict)
    
    # Save model
    predictor.save_model('models/gold_predictor_logistic.pkl')
    
    # Test prediction on latest data
    print("="*80)
    print("SAMPLE PREDICTION (most recent data)")
    print("="*80)
    
    latest_data = X.iloc[[-1]]
    predictions = predictor.predict_all_horizons(latest_data)
    
    for horizon, pred in predictions.items():
        print(f"\n{horizon.upper()}-term prediction:")
        print(f"  Direction: {pred['prediction']}")
        print(f"  Probability UP:   {pred['probability_up']:.1%}")
        print(f"  Probability DOWN: {pred['probability_down']:.1%}")
        print(f"  Confidence: {pred['confidence']:.1%}")

