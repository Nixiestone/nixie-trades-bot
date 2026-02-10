"""
ml_models.py - Machine Learning Models for Nix Trades
LSTM + XGBoost Ensemble with Rule-Based Fallback

Models:
- LSTM (70% weight): Sequence-based prediction on OHLCV data
- XGBoost (30% weight): Feature-based classification

Fallback: If models unavailable, use rule-based SMC scoring

NO EMOJIS - Professional code only
"""

import logging
import os
import pickle
from typing import Optional, Dict, Tuple
import numpy as np
import config

logger = logging.getLogger(__name__)

# Optional ML imports (graceful degradation if not available)
try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM as LSTM_Layer, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available - using rule-based fallback")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available - using rule-based fallback")


class MLEnsemble:
    """
    ML Ensemble combining LSTM and XGBoost for setup confidence scoring.
    Falls back to rule-based if models unavailable.
    """
    
    def __init__(self, models_dir: str = './models'):
        """
        Initialize ML ensemble.
        
        Args:
            models_dir: Directory containing saved model files
        """
        self.models_dir = models_dir
        self.lstm_model = None
        self.xgboost_model = None
        self.models_loaded = False
        
        # Try to load existing models
        self._load_models()
    
    
    def _load_models(self) -> None:
        """
        Load pre-trained LSTM and XGBoost models from disk.
        """
        try:
            # Load LSTM
            if TENSORFLOW_AVAILABLE:
                lstm_path = os.path.join(self.models_dir, 'lstm_model.h5')
                if os.path.exists(lstm_path):
                    self.lstm_model = load_model(lstm_path)
                    logger.info("LSTM model loaded successfully")
                else:
                    logger.warning(f"LSTM model not found at {lstm_path}")
            
            # Load XGBoost
            if XGBOOST_AVAILABLE:
                xgb_path = os.path.join(self.models_dir, 'xgboost_model.pkl')
                if os.path.exists(xgb_path):
                    with open(xgb_path, 'rb') as f:
                        self.xgboost_model = pickle.load(f)
                    logger.info("XGBoost model loaded successfully")
                else:
                    logger.warning(f"XGBoost model not found at {xgb_path}")
            
            # Check if at least one model loaded
            self.models_loaded = (self.lstm_model is not None or self.xgboost_model is not None)
            
            if not self.models_loaded:
                logger.warning("No ML models loaded - will use rule-based fallback")
        
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            self.models_loaded = False
    
    
    def predict(
        self,
        ohlcv_sequence: np.ndarray,
        features: Dict[str, float]
    ) -> Tuple[int, int, int]:
        """
        Generate ML predictions for setup.
        
        Args:
            ohlcv_sequence: OHLCV data (last 100 candles) for LSTM
            features: Feature dict for XGBoost
            
        Returns:
            Tuple[int, int, int]: (ml_score, lstm_score, xgboost_score)
            All scores 0-100
        """
        try:
            # If no models loaded, use rule-based
            if not self.models_loaded:
                return self._rule_based_scoring(features)
            
            lstm_score = 0
            xgboost_score = 0
            
            # LSTM Prediction
            if self.lstm_model is not None and ohlcv_sequence is not None:
                lstm_score = self._predict_lstm(ohlcv_sequence)
            else:
                # Fallback to rule-based for LSTM
                lstm_score = self._rule_based_lstm_fallback(features)
            
            # XGBoost Prediction
            if self.xgboost_model is not None:
                xgboost_score = self._predict_xgboost(features)
            else:
                # Fallback to rule-based for XGBoost
                xgboost_score = self._rule_based_xgboost_fallback(features)
            
            # Weighted ensemble (70% LSTM, 30% XGBoost)
            ml_score = int(
                (lstm_score * config.ML_LSTM_WEIGHT) +
                (xgboost_score * config.ML_XGBOOST_WEIGHT)
            )
            
            return ml_score, lstm_score, xgboost_score
        
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            # Fallback on error
            return self._rule_based_scoring(features)
    
    
    def _predict_lstm(self, ohlcv_sequence: np.ndarray) -> int:
        """
        LSTM prediction on OHLCV sequence.
        
        Args:
            ohlcv_sequence: Shape (100, 5) - last 100 candles [OHLCV]
            
        Returns:
            int: LSTM confidence score (0-100)
        """
        try:
            # Reshape for LSTM input: (batch_size, timesteps, features)
            sequence = ohlcv_sequence.reshape(1, config.ML_SEQUENCE_LENGTH, 5)
            
            # Normalize (simple min-max for demo, use proper scaler in production)
            sequence = (sequence - sequence.min()) / (sequence.max() - sequence.min() + 1e-8)
            
            # Predict: returns [bullish_prob, bearish_prob, neutral_prob]
            predictions = self.lstm_model.predict(sequence, verbose=0)[0]
            
            # Take max confidence
            max_confidence = float(np.max(predictions))
            
            # Convert to 0-100 scale
            score = int(max_confidence * 100)
            
            return min(score, 100)
        
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {e}")
            return 50  # Neutral fallback
    
    
    def _predict_xgboost(self, features: Dict[str, float]) -> int:
        """
        XGBoost prediction on engineered features.
        
        Args:
            features: Dict of engineered features
            
        Returns:
            int: XGBoost confidence score (0-100)
        """
        try:
            # Convert features to array
            # (In production, feature order must match training)
            feature_array = np.array([
                features.get('htf_trend', 0),
                features.get('bos_count', 0),
                features.get('ob_strength', 0),
                features.get('fvg_size', 0),
                features.get('volume_ratio', 1.0),
                features.get('atr_ratio', 1.0),
                features.get('rr_ratio', 1.0),
                features.get('session_score', 0)
                # ... add more features as needed
            ]).reshape(1, -1)
            
            # Predict: returns probabilities [bearish, neutral, bullish]
            predictions = self.xgboost_model.predict_proba(feature_array)[0]
            
            # Take max confidence
            max_confidence = float(np.max(predictions))
            
            # Convert to 0-100 scale
            score = int(max_confidence * 100)
            
            return min(score, 100)
        
        except Exception as e:
            logger.error(f"Error in XGBoost prediction: {e}")
            return 50  # Neutral fallback
    
    
    # ==================== RULE-BASED FALLBACKS ====================
    
    def _rule_based_scoring(self, features: Dict[str, float]) -> Tuple[int, int, int]:
        """
        Pure rule-based scoring when no ML models available.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Tuple[int, int, int]: (ml_score, lstm_score, xgboost_score)
        """
        try:
            score = 50  # Base score
            
            # HTF trend alignment: +10
            if features.get('htf_trend') == features.get('expected_direction'):
                score += 10
            
            # BOS count: +5 per BOS (max +15)
            score += min(features.get('bos_count', 0) * 5, 15)
            
            # Volume confirmation: +10
            if features.get('volume_ratio', 0) >= 1.5:
                score += 10
            
            # FVG present (Unicorn): +15
            if features.get('fvg_size', 0) > 0:
                score += 15
            
            # Good R:R: +10
            if features.get('rr_ratio', 0) >= 2.0:
                score += 10
            
            # Strong session (London/NY): +5
            if features.get('session_score', 0) >= 8:
                score += 5
            
            # Normal volatility: +5
            atr_ratio = features.get('atr_ratio', 1.0)
            if 0.7 <= atr_ratio <= 2.0:
                score += 5
            
            # Cap at 100
            score = min(score, 100)
            
            # Return same score for all (no ML differentiation)
            return score, score, score
        
        except Exception as e:
            logger.error(f"Error in rule-based scoring: {e}")
            return 60, 60, 60  # Default moderate confidence
    
    
    def _rule_based_lstm_fallback(self, features: Dict[str, float]) -> int:
        """
        Rule-based fallback for LSTM score.
        """
        score = 50
        
        # Momentum-based scoring (simulating LSTM's sequence analysis)
        if features.get('recent_momentum', 0) > 0.5:
            score += 20
        
        # Trend strength
        if features.get('trend_strength', 0) > 0.7:
            score += 15
        
        # Volume trend
        if features.get('volume_increasing', False):
            score += 15
        
        return min(score, 100)
    
    
    def _rule_based_xgboost_fallback(self, features: Dict[str, float]) -> int:
        """
        Rule-based fallback for XGBoost score.
        """
        score = 50
        
        # Feature-based scoring (simulating XGBoost's decision tree logic)
        if features.get('ob_strength', 0) > 0.8:
            score += 15
        
        if features.get('fvg_size', 0) > 10:  # > 10 pips
            score += 10
        
        if features.get('rr_ratio', 0) >= 2.0:
            score += 10
        
        if features.get('volume_ratio', 0) >= 1.8:
            score += 10
        
        if features.get('bos_count', 0) >= 2:
            score += 5
        
        return min(score, 100)
    
    
    # ==================== MODEL TRAINING (Placeholder) ====================
    
    def train_lstm(self, training_data: np.ndarray, labels: np.ndarray) -> bool:
        """
        Train LSTM model on historical data.
        
        Args:
            training_data: OHLCV sequences, shape (n_samples, 100, 5)
            labels: Binary labels (0=bearish, 1=bullish), shape (n_samples,)
            
        Returns:
            bool: True if training successful
            
        Note:
            This is a placeholder. Actual training happens in ml_training.py
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available - cannot train LSTM")
            return False
        
        try:
            logger.info("Building LSTM model...")
            
            model = Sequential([
                LSTM_Layer(128, return_sequences=True, input_shape=(100, 5)),
                Dropout(0.2),
                LSTM_Layer(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(3, activation='softmax')  # 3 classes: bearish, neutral, bullish
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Training LSTM model...")
            model.fit(
                training_data,
                labels,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            
            # Save model
            os.makedirs(self.models_dir, exist_ok=True)
            model_path = os.path.join(self.models_dir, 'lstm_model.h5')
            model.save(model_path)
            logger.info(f"LSTM model saved to {model_path}")
            
            self.lstm_model = model
            return True
        
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            return False
    
    
    def train_xgboost(self, training_features: np.ndarray, labels: np.ndarray) -> bool:
        """
        Train XGBoost model on engineered features.
        
        Args:
            training_features: Feature array, shape (n_samples, n_features)
            labels: Binary labels (0=bearish, 1=bullish)
            
        Returns:
            bool: True if training successful
        """
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost not available - cannot train")
            return False
        
        try:
            logger.info("Training XGBoost model...")
            
            model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.01,
                n_estimators=500,
                objective='multi:softprob',
                num_class=3  # bearish, neutral, bullish
            )
            
            model.fit(training_features, labels)
            
            # Save model
            os.makedirs(self.models_dir, exist_ok=True)
            model_path = os.path.join(self.models_dir, 'xgboost_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"XGBoost model saved to {model_path}")
            
            self.xgboost_model = model
            return True
        
        except Exception as e:
            logger.error(f"Error training XGBoost: {e}")
            return False