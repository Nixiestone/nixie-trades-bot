"""
NIX TRADES - Machine Learning Ensemble  (Part 1 of 2)
Role: Senior Quantitative Developer + Data Scientist

KEY CHANGE from previous version:
  Training now uses the REAL SMCStrategy class for POI detection.
  Previously _generate_training_samples() had its own hand-rolled POI
  finder. That meant training data came from different logic than live
  signals, so models learned patterns the live system would never see.

  Now the training pipeline mirrors the live pipeline exactly:
    Step 1: smc.determine_htf_trend(d1_window)   - same as scheduler
    Step 2: smc.detect_break_of_structure(h1_window, direction)
            smc.detect_market_structure_shift(h1_window, direction)
            smc.detect_breaker_blocks(h1_window, ...)  - same as scheduler
            smc.detect_order_blocks(h1_window, direction) - same as scheduler
    Step 3: smc.calculate_entry_price(poi, ...) + calculate_stop_loss()
            to get real entry and SL for WIN/LOSS labeling
    Step 4: extract_features(window, poi, htf_trend, setup_type) - same fn
            called by both training AND live prediction

  The result: models learn from the exact same setups that the live bot
  would send, making predictions genuinely predictive.

Disk persistence, auto-retrain, ensemble scoring: all unchanged.

NO EMOJIS - Enterprise code only
NO PLACEHOLDERS - All logic complete
"""

import logging
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import config
import utils

logger = logging.getLogger(__name__)

_MODEL_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
_XGB_PATH    = os.path.join(_MODEL_DIR, 'xgboost_model.pkl')
_LSTM_PATH   = os.path.join(_MODEL_DIR, 'lstm_model.pkl')
_SCALER_PATH = os.path.join(_MODEL_DIR, 'scaler.pkl')
_META_PATH   = os.path.join(_MODEL_DIR, 'training_metadata.pkl')
_FEATURE_DIM = 22


class MLEnsemble:
    """
    ML ensemble: XGBoost + GradientBoostingClassifier.
    Loads trained models from disk on startup.
    Training uses the real SMCStrategy so training data = live signal data.
    """

    def __init__(self, mt5_connector=None):
        self.logger         = logging.getLogger(f"{__name__}.MLEnsemble")
        self.mt5            = mt5_connector
        self.xgboost_model  = None
        self.lstm_model     = None
        self.scaler         = StandardScaler()
        self.models_trained = False
        self.training_metadata: Dict = {}

        # Lazy-load SMC strategy to avoid circular import at module level
        self._smc = None

        # Live outcome accumulator for auto-retrain every 100 trades
        self.setups_since_training: int  = 0
        self.training_threshold:    int  = 100
        self.training_data_history: List[Tuple[np.ndarray, float]] = []
        self.max_history_size:      int  = 2000

        os.makedirs(_MODEL_DIR, exist_ok=True)
        self._load_from_disk()
        self.logger.info("ML Ensemble initialised. Models trained: %s", self.models_trained)

    @property
    def smc(self):
        """Lazy-load SMCStrategy to avoid circular import."""
        if self._smc is None:
            try:
                from smc_strategy import SMCStrategy
                self._smc = SMCStrategy()
                self.logger.info("SMCStrategy loaded into ML Ensemble for training.")
            except Exception as e:
                self.logger.error("Could not load SMCStrategy: %s", e)
                self._smc = None
        return self._smc

    # ==================== DISK PERSISTENCE ====================

    def _load_from_disk(self) -> bool:
        try:
            if not all(os.path.exists(p) for p in [_XGB_PATH, _LSTM_PATH, _SCALER_PATH]):
                self.logger.info(
                    "No saved models found. Run train_models.py once before going live.")
                return False
            with open(_XGB_PATH,    'rb') as f: self.xgboost_model = pickle.load(f)
            with open(_LSTM_PATH,   'rb') as f: self.lstm_model    = pickle.load(f)
            with open(_SCALER_PATH, 'rb') as f: self.scaler        = pickle.load(f)
            if os.path.exists(_META_PATH):
                with open(_META_PATH, 'rb') as f: self.training_metadata = pickle.load(f)
            self.models_trained = True
            self.logger.info(
                "Trained models loaded. Samples: %s. XGBoost accuracy: %s.",
                self.training_metadata.get('samples', 'N/A'),
                self.training_metadata.get('xgboost_accuracy', 'N/A'))
            return True
        except Exception as e:
            self.logger.error("Failed to load models from disk: %s", e)
            self.models_trained = False
            return False

    def _save_to_disk(self) -> bool:
        try:
            with open(_XGB_PATH,    'wb') as f: pickle.dump(self.xgboost_model,    f)
            with open(_LSTM_PATH,   'wb') as f: pickle.dump(self.lstm_model,        f)
            with open(_SCALER_PATH, 'wb') as f: pickle.dump(self.scaler,            f)
            with open(_META_PATH,   'wb') as f: pickle.dump(self.training_metadata, f)
            self.logger.info("Models saved to disk at %s", _MODEL_DIR)
            return True
        except Exception as e:
            self.logger.error("Failed to save models: %s", e)
            return False

    # ==================== HISTORICAL TRAINING ====================

    def train_on_historical_data(
        self,
        symbols:    List[str] = None,
        start_date: str = '2020-01-01',
        end_date:   str = '2026-02-24',
    ) -> bool:
        """
        Train both models on MT5 historical data using the real SMC strategy.
        Called ONCE by train_models.py before going live.
        """
        if symbols is None:
            symbols = list(config.MONITORED_SYMBOLS)

        if not self.mt5:
            self.logger.error("Cannot train: no mt5_connector provided.")
            return False
        if not self.mt5.is_worker_reachable():
            self.logger.error("Cannot train: MT5 worker not reachable.")
            return False
        if self.smc is None:
            self.logger.error("Cannot train: SMCStrategy failed to load.")
            return False

        self.logger.info(
            "Starting historical training on %d symbols using SMC strategy. "
            "Period: %s to %s.", len(symbols), start_date, end_date)

        all_features: List[np.ndarray] = []
        all_labels:   List[float]      = []

        for symbol in symbols:
            self.logger.info("Fetching data for %s ...", symbol)
            try:
                # Fetch within broker limits. Most brokers allow 10,000-50,000
                # bars per request. 50,000 M15 bars = ~2 years of data.
                m15_raw = self.mt5.get_historical_data(symbol, 'M15', bars=50000)
                h1_raw  = self.mt5.get_historical_data(symbol, 'H1',  bars=10000)
                d1_raw  = self.mt5.get_historical_data(symbol, 'D1',  bars=1500)

                if not m15_raw or len(m15_raw) < 500:
                    self.logger.warning(
                        "Insufficient M15 data for %s (%d bars). Skipping.",
                        symbol, len(m15_raw) if m15_raw else 0)
                    continue

                def to_df(raw):
                    df = pd.DataFrame(raw)
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                    df.sort_index(inplace=True)
                    return df

                feats, labels = self._generate_training_samples(
                    symbol, to_df(m15_raw), to_df(h1_raw), to_df(d1_raw))
                self.logger.info(
                    "%s: %d labeled samples from %d M15 bars.",
                    symbol, len(feats), len(m15_raw))
                all_features.extend(feats)
                all_labels.extend(labels)

            except Exception as e:
                self.logger.error("Error processing %s: %s", symbol, e)

        if len(all_features) < 200:
            self.logger.error(
                "Only %d samples generated. Need at least 200. "
                "Check MT5 worker connection and broker data availability.",
                len(all_features))
            return False

        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels,   dtype=np.float32)
        self.logger.info(
            "Total training samples: %d. Win rate in data: %.1f%%.",
            len(X), y.mean() * 100)

        success = self._train_both_models(X, y)
        if success:
            seed_n = min(len(all_features), self.max_history_size // 2)
            for f, l in zip(all_features[-seed_n:], all_labels[-seed_n:]):
                self.training_data_history.append((f, l))
        return success

    def _generate_training_samples(
        self, symbol: str,
        m15_df: pd.DataFrame,
        h1_df:  pd.DataFrame,
        d1_df:  pd.DataFrame,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Slide a window across M15 history.

        At each step, use the REAL SMC strategy (same as live trading) to:
          1. Determine D1 trend context
          2. Detect BOS/MSS on H1
          3. Identify the POI (Order Block or Breaker Block)
          4. Extract 22-element feature vector
          5. Label WIN/LOSS by looking 40 bars forward for 1:2 RR

        Inconclusive windows (neither target nor SL reached) are discarded.
        This means training data is a realistic sample of real setups.
        """
        feats:  List[np.ndarray] = []
        labels: List[float]      = []
        window_size  = 100    # M15 bars of context per window
        forward_bars = 40     # Bars to look ahead for labeling
        step         = 15     # Step between windows (larger = faster training)

        try:
            for i in range(window_size, len(m15_df) - forward_bars, step):
                cur_time = m15_df.index[i]

                # Build time-bounded context slices
                d1_ctx = d1_df[d1_df.index <= cur_time].tail(50)
                h1_ctx = h1_df[h1_df.index <= cur_time].tail(150)
                m15_win = m15_df.iloc[i - window_size: i].copy()

                if len(d1_ctx) < 20 or len(h1_ctx) < 50 or len(m15_win) < window_size:
                    continue

                # --- Phase 1: HTF trend (D1) via real SMC ---
                try:
                    htf_trend = self.smc.determine_htf_trend(d1_ctx)
                except Exception:
                    continue

                if htf_trend.get('trend') == 'RANGING':
                    continue

                direction  = htf_trend['trend']  # 'BULLISH' or 'BEARISH'
                smc_dir    = direction            # alias for clarity

                # --- Phase 2: Structure detection (H1) via real SMC ---
                setup_type = None
                poi        = None

                try:
                    bos_events = self.smc.detect_break_of_structure(h1_ctx, smc_dir)
                    mss_event  = self.smc.detect_market_structure_shift(h1_ctx, smc_dir)

                    if len(bos_events) >= 2:
                        setup_type = 'BOS'
                        breakers   = self.smc.detect_breaker_blocks(
                            h1_ctx, smc_dir,
                            htf_trend.get('swing_high', 0),
                            htf_trend.get('swing_low', 0))
                        if breakers:
                            poi = breakers[0]
                        else:
                            # Fall back to Order Block if no breaker found
                            obs = self.smc.detect_order_blocks(h1_ctx, smc_dir)
                            if obs:
                                poi = obs[0]
                    elif mss_event:
                        setup_type = 'MSS'
                        obs = self.smc.detect_order_blocks(h1_ctx, smc_dir)
                        if obs:
                            poi = obs[0]
                except Exception:
                    continue

                if poi is None or setup_type is None:
                    continue

                # --- Phase 3: Entry/SL via real SMC ---
                try:
                    atr_val    = self.smc._calculate_atr(m15_win.tail(20))
                    entry_cfg  = self.smc.calculate_entry_price(poi, 'STANDARD', 60)
                    sl_cfg     = self.smc.calculate_stop_loss(
                        poi, smc_dir, symbol, atr_val)
                    entry      = float(entry_cfg['entry_price'])
                    sl         = float(sl_cfg['stop_loss'])
                except Exception:
                    continue

                risk = abs(entry - sl)
                if risk < 1e-9:
                    continue

                # --- Phase 4: Feature extraction (exact same fn as live) ---
                try:
                    f = self.extract_features(m15_win, poi, htf_trend, setup_type)
                except Exception:
                    continue

                # --- Phase 5: Label WIN/LOSS from future bars ---
                future = m15_df.iloc[i: i + forward_bars]
                if len(future) < forward_bars:
                    continue

                target_reward = risk * 2.0   # 1:2 RR target
                is_buy = direction == 'BULLISH'

                if is_buy:
                    won     = future['high'].max() >= (entry + target_reward)
                    knocked = future['low'].min()  <= sl
                else:
                    won     = future['low'].min()  <= (entry - target_reward)
                    knocked = future['high'].max() >= sl

                if won and not knocked:
                    label = 1.0
                elif knocked:
                    label = 0.0
                else:
                    continue   # Inconclusive - discard

                feats.append(f)
                labels.append(label)

        except Exception as e:
            self.logger.error("Sample generation error for %s: %s", symbol, e)

        return feats, labels

    # ==================== MODEL TRAINING ====================

    def _train_both_models(self, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
            self.scaler  = StandardScaler()
            X_tr_s = self.scaler.fit_transform(X_tr)
            X_te_s = self.scaler.transform(X_te)
            ok1    = self._fit_xgboost(X_tr_s, y_tr, X_te_s, y_te)
            ok2    = self._fit_lstm_sub(X_tr_s, y_tr, X_te_s, y_te)
            if not (ok1 and ok2):
                return False
            self.models_trained = True
            return self._save_to_disk()
        except Exception as e:
            self.logger.error("Training error: %s", e, exc_info=True)
            return False

    def _fit_xgboost(self, X_tr, y_tr, X_te, y_te) -> bool:
        try:
            import xgboost as xgb
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dtest  = xgb.DMatrix(X_te, label=y_te)
            params = {
                'objective': 'binary:logistic', 'max_depth': 6,
                'learning_rate': 0.05, 'subsample': 0.8,
                'colsample_bytree': 0.8, 'min_child_weight': 3,
                'gamma': 0.1, 'eval_metric': 'auc', 'seed': 42,
            }
            self.xgboost_model = xgb.train(
                params, dtrain, num_boost_round=400,
                evals=[(dtest, 'test')], early_stopping_rounds=30,
                verbose_eval=False)
            y_prob = self.xgboost_model.predict(dtest)
            acc = accuracy_score(y_te, (y_prob > 0.5).astype(int))
            auc = roc_auc_score(y_te, y_prob)
        except ImportError:
            self.logger.warning(
                "XGBoost not installed. Using sklearn GBC substitute. "
                "Install with: pip install xgboost --break-system-packages")
            self.xgboost_model = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.08,
                subsample=0.8, random_state=42)
            self.xgboost_model.fit(X_tr, y_tr)
            acc = accuracy_score(y_te, self.xgboost_model.predict(X_te))
            auc = 0.0
        except Exception as e:
            self.logger.error("XGBoost training failed: %s", e)
            return False

        self.logger.info(
            "XGBoost training complete. Accuracy: %.1f%%  AUC: %.3f  "
            "Train: %d  Test: %d", acc * 100, auc, len(X_tr), len(X_te))
        self.training_metadata.update({
            'xgboost_accuracy': f"{acc*100:.1f}%",
            'xgboost_auc':      f"{auc:.3f}",
            'samples':          len(X_tr) + len(X_te),
            'trained_at':       datetime.now(timezone.utc).isoformat()})
        return True

    def _fit_lstm_sub(self, X_tr, y_tr, X_te, y_te) -> bool:
        """GradientBoostingClassifier with different hyperparameters for ensemble diversity."""
        try:
            self.lstm_model = GradientBoostingClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.06,
                subsample=0.75, min_samples_leaf=10, random_state=99)
            self.lstm_model.fit(X_tr, y_tr)
            y_prob = self.lstm_model.predict_proba(X_te)[:, 1]
            acc = accuracy_score(y_te, (y_prob > 0.5).astype(int))
            auc = roc_auc_score(y_te, y_prob)
            self.logger.info(
                "LSTM substitute training complete. Accuracy: %.1f%%  AUC: %.3f",
                acc * 100, auc)
            self.training_metadata.update({
                'lstm_accuracy': f"{acc*100:.1f}%",
                'lstm_auc':      f"{auc:.3f}"})
            return True
        except Exception as e:
            self.logger.error("LSTM substitute training failed: %s", e)
            return False

    # ==================== LIVE AUTO-RETRAIN ====================

    def record_trade_outcome(self, features: np.ndarray, won: bool) -> bool:
        """
        Record a live trade outcome. Called by position_monitor after each close.
        Every 100 outcomes, both models retrain on the combined dataset.

        Args:
            features: 22-element vector from get_ensemble_prediction()['features']
            won:      True = TP2 hit. False = stop loss hit.

        Returns:
            True if a retrain was triggered and succeeded.
        """
        self.training_data_history.append((features, 1.0 if won else 0.0))
        if len(self.training_data_history) > self.max_history_size:
            self.training_data_history = self.training_data_history[-self.max_history_size:]

        self.setups_since_training += 1

        if self.setups_since_training >= self.training_threshold:
            self.logger.info(
                "Auto-retrain triggered: %d live outcomes accumulated.",
                self.training_threshold)
            self.setups_since_training = 0
            X = np.array([h[0] for h in self.training_data_history], dtype=np.float32)
            y = np.array([h[1] for h in self.training_data_history], dtype=np.float32)
            if len(X) >= 200:
                success = self._train_both_models(X, y)
                if success:
                    self.logger.info(
                        "Auto-retrain complete. Total samples: %d.", len(X))
                return success
            else:
                self.logger.warning(
                    "Auto-retrain skipped: only %d samples (need 200+).", len(X))
        return False

    # ==================== FEATURE EXTRACTION ====================

    def extract_features(
        self, data: pd.DataFrame, poi: Dict, htf_trend: Dict, setup_type: str
    ) -> np.ndarray:
        """
        Convert OHLCV data and POI into a 22-element normalised feature vector.

        This SAME function is called during:
          - Training: _generate_training_samples() uses this
          - Live use: predict_lstm(), predict_xgboost(), get_ensemble_prediction()

        That consistency is the core reason the models are predictive.

        Index  Feature
          0    HTF alignment (1 = POI direction matches D1 trend)
          1    RSI normalised 0-1
          2    RSI in favorable zone (oversold for BUY, overbought for SELL)
          3    ATR / price (volatility measure)
          4    ATR ratio recent vs average (normalised, capped 5x)
          5    MACD agrees with direction
          6    POI candle volume vs average (normalised, capped 5x)
          7    Recent 5-bar volume surge (normalised, capped 5x)
          8    Impulse move size in pips (normalised)
          9    POI freshness: 1 / (1 + bars since POI)
         10    Close / upper Bollinger Band
         11    Close / lower Bollinger Band
         12    Price on favorable side of 50-bar SMA
         13    Price on favorable side of 20-bar SMA
         14    POI is Order Block (1 or 0)
         15    POI is Breaker Block (1 or 0)
         16    POI is Fair Value Gap (1 or 0)
         17    Setup type is BOS (1) or MSS (0)
         18    Last candle body / total range
         19    Close vs POI midpoint (direction-aware)
         20    EMA9 vs EMA21 momentum agrees with direction
         21    Consecutive same-direction closes / 10
        """
        try:
            close  = data['close'].values
            high   = data['high'].values
            low    = data['low'].values
            open_  = data['open'].values
            volume = (data['volume'].values if 'volume' in data.columns
                      else np.ones(len(data)))

            direction = poi.get('direction', 'BULLISH')
            is_buy    = direction == 'BULLISH'
            price     = float(close[-1])

            # 0: HTF alignment
            htf_ok = 1.0 if htf_trend.get('trend') == direction else 0.0

            # 1-2: RSI
            rsi      = self._calc_rsi(data)
            rsi_zone = (1.0 if (is_buy and 30 <= rsi <= 55)
                        or (not is_buy and 45 <= rsi <= 70) else 0.0)

            # 3-4: ATR
            atr      = self._calc_atr(data, 14)
            atr_norm = atr / price if price > 0 else 0.0
            avg_diff = float(data['close'].diff().abs().mean())
            atr_r    = min((atr / avg_diff) if avg_diff > 0 else 1.0, 5.0) / 5.0

            # 5: MACD
            mv, sv   = self._calc_macd(data)
            macd_ok  = 1.0 if (is_buy and mv > sv) or (not is_buy and mv < sv) else 0.0

            # 6-8: Volume & impulse
            avg_vol  = float(volume.mean()) if volume.mean() > 0 else 1.0
            poi_vr   = min(float(poi.get('volume_ratio', 1.0)), 5.0) / 5.0
            rec_vol  = float(volume[-5:].mean()) if len(volume) >= 5 else avg_vol
            vol_sg   = min(rec_vol / avg_vol, 5.0) / 5.0
            imp_p    = min(float(poi.get('impulse_pips', 0)) / 100.0, 2.0)

            # 9: Freshness
            bars_ago = max(0, len(data) - int(poi.get('index', len(data) - 1)))
            fresh    = 1.0 / (1.0 + bars_ago)

            # 10-11: Bollinger Bands
            bb_up, bb_lo, _ = self._calc_bollinger(data)
            bb_up_r = price / (bb_up + 1e-9)
            bb_lo_r = price / (bb_lo + 1e-9)

            # 12-13: SMA filters
            def _sma(n):
                v = float(data['close'].rolling(n).mean().iloc[-1]) if len(data) >= n else price
                return price if np.isnan(v) else v

            sma50 = _sma(50)
            sma20 = _sma(20)
            ab50  = 1.0 if (is_buy and price > sma50) or (not is_buy and price < sma50) else 0.0
            ab20  = 1.0 if (is_buy and price > sma20) or (not is_buy and price < sma20) else 0.0

            # 14-16: POI type flags
            pt     = str(poi.get('type', 'OB')).upper()
            is_ob  = 1.0 if pt == 'OB'  else 0.0
            is_bb  = 1.0 if pt == 'BB'  else 0.0
            is_fvg = 1.0 if pt == 'FVG' else 0.0

            # 17: Setup type
            is_bos = 1.0 if setup_type == 'BOS' else 0.0

            # 18: Candle body ratio
            body_sz = abs(float(close[-1]) - float(open_[-1]))
            rng_sz  = max(float(high[-1]) - float(low[-1]), 1e-9)
            body_r  = body_sz / rng_sz

            # 19: Close vs POI midpoint
            poi_mid = (float(poi.get('high', price)) + float(poi.get('low', price))) / 2.0
            cpoi    = 1.0 if (is_buy and price < poi_mid) or (not is_buy and price > poi_mid) else 0.0

            # 20: EMA momentum
            ema9  = float(data['close'].ewm(span=9,  adjust=False).mean().iloc[-1])
            ema21 = float(data['close'].ewm(span=21, adjust=False).mean().iloc[-1])
            ema_ok = 1.0 if (is_buy and ema9 > ema21) or (not is_buy and ema9 < ema21) else 0.0

            # 21: Consecutive same-direction closes
            cnt = 0
            for k in range(len(close) - 1, max(len(close) - 11, 0), -1):
                if is_buy  and close[k] > open_[k]: cnt += 1
                elif not is_buy and close[k] < open_[k]: cnt += 1
                else: break

            f = np.array([
                htf_ok, rsi / 100.0, rsi_zone, atr_norm, atr_r,
                macd_ok, poi_vr, vol_sg, imp_p, fresh,
                bb_up_r, bb_lo_r, ab50, ab20,
                is_ob, is_bb, is_fvg, is_bos, body_r, cpoi,
                ema_ok, cnt / 10.0,
            ], dtype=np.float32)
            return np.nan_to_num(f, nan=0.0, posinf=1.0, neginf=0.0)

        except Exception as e:
            self.logger.error("Feature extraction error: %s", e)
            return np.zeros(_FEATURE_DIM, dtype=np.float32)

    # ==================== LIVE PREDICTIONS ====================

    def predict_lstm(self, data, poi, htf_trend, setup_type) -> int:
        try:
            f = self.extract_features(data, poi, htf_trend, setup_type)
            if self.models_trained and self.lstm_model is not None:
                p = self.lstm_model.predict_proba(
                    self.scaler.transform(f.reshape(1, -1)))[0][1]
                return int(p * 100)
            return self._heuristic_lstm(data, poi, htf_trend)
        except Exception as e:
            self.logger.error("LSTM prediction error: %s", e)
            return 50

    def predict_xgboost(self, data, poi, htf_trend, setup_type) -> int:
        try:
            f   = self.extract_features(data, poi, htf_trend, setup_type)
            f_s = self.scaler.transform(f.reshape(1, -1))
            if self.models_trained and self.xgboost_model is not None:
                try:
                    import xgboost as xgb
                    p = self.xgboost_model.predict(xgb.DMatrix(f_s))[0]
                except Exception:
                    p = self.xgboost_model.predict_proba(f_s)[0][1]
                return int(p * 100)
            return self._heuristic_xgboost(data, poi, htf_trend, setup_type)
        except Exception as e:
            self.logger.debug(
                "XGBoost not trained yet, using heuristic fallback (50%%): %s", e)
            return 50

    def get_ensemble_prediction(self, data, poi, htf_trend, setup_type) -> Dict:
        """
        Combined prediction. XGBoost weight 60%, LSTM weight 40%.
        The 'features' key must be saved and passed to record_trade_outcome()
        when the trade closes, so the model learns from this specific trade.
        """
        try:
            features  = self.extract_features(data, poi, htf_trend, setup_type)
            lstm_s    = self.predict_lstm(data, poi, htf_trend, setup_type)
            xgb_s     = self.predict_xgboost(data, poi, htf_trend, setup_type)
            consensus = int(lstm_s * 0.4 + xgb_s * 0.6)
            diff      = abs(lstm_s - xgb_s)
            agreement = 'STRONG' if diff <= 10 else ('MODERATE' if diff <= 20 else 'WEAK')
            self.logger.info(
                "Ensemble: LSTM=%d%%  XGBoost=%d%%  Consensus=%d%%  "
                "Agreement=%s  Trained: %s",
                lstm_s, xgb_s, consensus, agreement, self.models_trained)
            return {
                'lstm_score': lstm_s, 'xgboost_score': xgb_s,
                'consensus_score': consensus, 'agreement': agreement,
                'direction': poi.get('direction', 'BULLISH'), 'features': features,
            }
        except Exception as e:
            self.logger.error("Ensemble prediction error: %s", e)
            return {
                'lstm_score': 50, 'xgboost_score': 50, 'consensus_score': 50,
                'agreement': 'WEAK', 'direction': poi.get('direction', 'BULLISH'),
                'features': np.zeros(_FEATURE_DIM, dtype=np.float32),
            }

    def should_send_setup(self, consensus_score: int) -> Tuple[bool, str]:
        """Return (should_send, tier_name) based on consensus score."""
        if consensus_score >= config.ML_TIER_PREMIUM:         return True, 'PREMIUM'
        elif consensus_score >= config.ML_TIER_STANDARD:      return True, 'STANDARD'
        elif consensus_score >= config.ML_TIER_DISCRETIONARY: return True, 'DISCRETIONARY'
        return False, 'REJECTED'

    def should_auto_execute(self, consensus_score: int) -> bool:
        return consensus_score >= config.ML_AUTO_EXECUTE_THRESHOLD

    def get_model_status(self) -> dict:
        """Return model status dict for the /status command."""
        if not self.models_trained:
            return {
                'trained':     False,
                'status_text': (
                    'Models not yet trained. Using calibrated heuristics. '
                    'Run: python train_models.py')}
        return {
            'trained':          True,
            'status_text':      'Trained ML models active (trained on SMC strategy data).',
            'xgboost_accuracy': self.training_metadata.get('xgboost_accuracy', 'N/A'),
            'lstm_accuracy':    self.training_metadata.get('lstm_accuracy',    'N/A'),
            'xgboost_auc':      self.training_metadata.get('xgboost_auc',      'N/A'),
            'training_date':    self.training_metadata.get('trained_at',       'Unknown'),
            'samples':          self.training_metadata.get('samples',           0),
            'live_outcomes_since_last_retrain': self.setups_since_training,
            'retrain_threshold': self.training_threshold,
        }

    # ==================== HEURISTIC FALLBACKS (PRE-TRAINING) ====================

    def _heuristic_lstm(self, data, poi, htf_trend) -> int:
        """Narrow range (45-72) to be honest about uncertainty."""
        score = 55
        try:
            is_buy = poi.get('direction', 'BULLISH') == 'BULLISH'
            if htf_trend.get('trend') == poi.get('direction'): score += 10
            rsi = self._calc_rsi(data)
            if is_buy and 35 <= rsi <= 52: score += 7
            elif not is_buy and 48 <= rsi <= 65: score += 7
            if poi.get('volume_ratio', 0) >= 1.5: score += 8
            if 'volume' in data.columns:
                avg = data['volume'].mean()
                if avg > 0 and data['volume'].values[-5:].mean() > avg * 1.2: score += 6
        except Exception: score = 55
        return max(45, min(score, 72))

    def _heuristic_xgboost(self, data, poi, htf_trend, setup_type) -> int:
        """Range 38-78 before training."""
        score = 50
        try:
            direction = poi.get('direction', 'BULLISH')
            is_buy    = direction == 'BULLISH'
            if htf_trend.get('trend') == direction: score += 16
            else: score -= 8
            if poi.get('volume_ratio', 0) >= 1.5: score += 10
            rsi = self._calc_rsi(data)
            if is_buy and 30 <= rsi <= 50: score += 9
            elif not is_buy and 50 <= rsi <= 70: score += 9
            atr = self._calc_atr(data, 14)
            avg_d = float(data['close'].diff().abs().mean())
            if avg_d > 0:
                r = atr / avg_d
                if 0.7 <= r <= 2.0: score += 8
                else: score -= 12
            if setup_type == 'BOS' and poi.get('type') in ('BB', 'BREAKER'): score += 6
            elif setup_type == 'MSS' and poi.get('type') in ('OB', 'ORDER_BLOCK'): score += 6
            bars_ago = len(data) - int(poi.get('index', len(data) - 1))
            if bars_ago <= 20: score += 5
        except Exception: score = 50
        return max(38, min(score, 78))

    # ==================== INDICATOR HELPERS ====================

    def _calc_rsi(self, data: pd.DataFrame, p: int = 14) -> float:
        try:
            d = data['close'].diff()
            g = d.where(d > 0, 0.0).rolling(p).mean()
            l = (-d.where(d < 0, 0.0)).rolling(p).mean()
            v = float((100 - 100 / (1 + g / (l + 1e-9))).iloc[-1])
            return v if not np.isnan(v) else 50.0
        except Exception: return 50.0

    def _calc_atr(self, data: pd.DataFrame, p: int = 14) -> float:
        try:
            h = data['high']; l = data['low']; pc = data['close'].shift(1)
            tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
            v  = float(tr.rolling(p).mean().iloc[-1])
            return v if not np.isnan(v) else 0.0
        except Exception: return 0.0

    def _calc_macd(self, data, fast=12, slow=26, sig=9) -> Tuple[float, float]:
        try:
            ef = data['close'].ewm(span=fast, adjust=False).mean()
            es = data['close'].ewm(span=slow, adjust=False).mean()
            m  = ef - es; s = m.ewm(span=sig, adjust=False).mean()
            mv = float(m.iloc[-1]); sv = float(s.iloc[-1])
            return (mv if not np.isnan(mv) else 0.0,
                    sv if not np.isnan(sv) else 0.0)
        except Exception: return 0.0, 0.0

    def _calc_bollinger(self, data, p=20, sd=2.0) -> Tuple[float, float, float]:
        try:
            sma = data['close'].rolling(p).mean()
            std = data['close'].rolling(p).std()
            px  = float(data['close'].iloc[-1])
            up  = float((sma + std * sd).iloc[-1])
            lo  = float((sma - std * sd).iloc[-1])
            mid = float(sma.iloc[-1])
            return (up  if not np.isnan(up)  else px,
                    lo  if not np.isnan(lo)  else px,
                    mid if not np.isnan(mid) else px)
        except Exception:
            p = float(data['close'].iloc[-1]); return p, p, p