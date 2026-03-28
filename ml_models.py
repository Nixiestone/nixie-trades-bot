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
_RF_PATH     = os.path.join(_MODEL_DIR, 'rf_model.pkl')
_META_PATH   = os.path.join(_MODEL_DIR, 'training_metadata.pkl')
_FEATURE_DIM = 22

# Only train on setups scoring >= this value. Weak setups (~50% win rate) dilute training.
TRAINING_MIN_QUALITY_SCORE = 35


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
        self.rf_model = None
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
            # RF model is optional for backward compatibility with older model bundles.
            if os.path.exists(_RF_PATH):
                with open(_RF_PATH, 'rb') as f:
                    self.rf_model = pickle.load(f)
            else:
                self.rf_model = None
                self.logger.info(
                    "RF model file not found. Using 2-model ensemble (XGBoost + LSTM).")
            if os.path.exists(_META_PATH):
                with open(_META_PATH, 'rb') as f: self.training_metadata = pickle.load(f)
            self.models_trained = True
            self.logger.info(
                "Trained models loaded. Samples: %s. XGBoost accuracy: %s. RF loaded: %s.",
                self.training_metadata.get('samples', 'N/A'),
                self.training_metadata.get('xgboost_accuracy', 'N/A'),
                self.rf_model is not None)
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
            if self.rf_model is not None:
                with open(_RF_PATH, 'wb') as f: pickle.dump(self.rf_model, f)
            elif os.path.exists(_RF_PATH):
                # Avoid loading stale RF artifacts that no longer match current scaler/data.
                os.remove(_RF_PATH)
            with open(_META_PATH,   'wb') as f: pickle.dump(self.training_metadata, f)
            self.logger.info("Models saved to disk at %s", _MODEL_DIR)
            return True
        except Exception as e:
            self.logger.error("Failed to save models: %s", e)
            return False

    # ==================== HISTORICAL TRAINING ====================

    @staticmethod
    def _candles_to_df(raw: List[dict]) -> pd.DataFrame:
        """
        Convert MT5 candle payload to a UTC-indexed DataFrame.
        MT5 worker returns Unix timestamps in seconds.
        """
        df = pd.DataFrame(raw)
        if 'time' not in df.columns:
            raise ValueError("Candle payload missing 'time' field.")

        ts_num = pd.to_numeric(df['time'], errors='coerce')
        if ts_num.notna().sum() >= max(1, int(len(df) * 0.8)):
            df['time'] = pd.to_datetime(ts_num, unit='s', utc=True, errors='coerce')
        else:
            df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')

        df.dropna(subset=['time'], inplace=True)
        if df.empty:
            raise ValueError("No valid candle timestamps after parsing.")

        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def _estimate_fetch_bars(start_date: str, end_date: str) -> Tuple[int, int, int]:
        """
        Estimate bars needed per timeframe from a date range.
        Adds warm-up/lookahead buffers and caps values to practical worker limits.
        """
        try:
            start_ts = pd.to_datetime(start_date, utc=True, errors='raise')
            end_ts = pd.to_datetime(end_date, utc=True, errors='raise')
            if end_ts < start_ts:
                start_ts, end_ts = end_ts, start_ts
            total_days = max(int((end_ts - start_ts).total_seconds() // 86400) + 1, 1)
        except Exception:
            # Fallback to ~2 years if date parsing fails.
            total_days = 730

        # M15: 252 trading days x 96 bars = 24,192 bars per year.
        # Cap at 80,000 (~2.3 years). All major retail MT5 brokers reliably
        # return this volume in a single request. Requesting more triggers
        # truncated or empty responses which silently discard training samples.
        # H1: cap at 20,000 (~2.3 years). D1: cap at 3,000 (~12 years).
        m15_bars = min(max(total_days * 96  + 500, 2000), 80000)
        h1_bars  = min(max(total_days * 24  + 300, 1000), 20000)
        d1_bars  = min(max(total_days       + 60,  300),   3000)
        return m15_bars, h1_bars, d1_bars

    def train_on_historical_data(
        self,
        symbols:    List[str] = None,
        start_date: str = '2020-01-01',
        end_date:   Optional[str] = None,
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
        if not self.mt5.is_service_reachable_sync():
            self.logger.error(
                "Cannot train: %s not reachable.",
                self.mt5.service_label(),
            )
            return False
        if self.smc is None:
            self.logger.error("Cannot train: SMCStrategy failed to load.")
            return False

        if end_date is None:
            from datetime import datetime, timezone as _tz
            end_date = datetime.now(_tz.utc).strftime('%Y-%m-%d')

        self.logger.info(
            "Starting historical training on %d symbols using SMC strategy. "
            "Period: %s to %s.", len(symbols), start_date, end_date)

        m15_bars, h1_bars, d1_bars = self._estimate_fetch_bars(start_date, end_date)
        self.logger.info(
            "Historical fetch plan per symbol: M15=%d H1=%d D1=%d bars "
            "(derived from requested period).",
            m15_bars, h1_bars, d1_bars
        )

        all_features: List[np.ndarray] = []
        all_labels:   List[float]      = []

        for symbol in symbols:
            self.logger.info("Fetching data for %s ...", symbol)
            try:
                m15_raw = self.mt5.get_historical_data_sync(symbol, 'M15', bars=m15_bars)
                h1_raw  = self.mt5.get_historical_data_sync(symbol, 'H1',  bars=h1_bars)
                d1_raw  = self.mt5.get_historical_data_sync(symbol, 'D1',  bars=d1_bars)

                if not m15_raw or len(m15_raw) < 500:
                    self.logger.warning(
                        "Insufficient M15 data for %s (%d bars). Skipping.",
                        symbol, len(m15_raw) if m15_raw else 0)
                    continue

                feats, labels = self._generate_training_samples(
                    symbol,
                    self._candles_to_df(m15_raw),
                    self._candles_to_df(h1_raw),
                    self._candles_to_df(d1_raw),
                )
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

        success = self._train_all_models(X, y)
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
        window_size  = 100
        forward_bars = 32     # 8 hours on M15 (4 bars/hour) - matches live H1 expiry
        # Step must be >= forward_bars to ensure non-overlapping labeling windows.
        # Using step=40 with forward_bars=32 means adjacent windows share future bars,
        # creating autocorrelation between labels that inflates test accuracy to 98%.
        # step=64 gives non-overlapping future windows and independent labels.
        step         = 64

        # Diagnostic counters - logged at end so you know where samples are lost
        _cnt_total       = 0
        _cnt_asian       = 0
        _cnt_no_ctx      = 0
        _cnt_ranging     = 0
        _cnt_no_poi      = 0
        _cnt_bad_entry   = 0
        _cnt_low_quality = 0
        _cnt_unfilled    = 0  # Limit order never reached entry price — no trade in live either
        _cnt_no_label    = 0  # Order filled but neither TP nor SL reached within expiry
        _cnt_labeled     = 0

        # Suppress INFO-level logs from SMC detection functions during training.
        # detect_break_of_structure and detect_market_structure_shift both log at
        # INFO level on every window call, producing thousands of log lines that
        # belong to the training loop, not to live signal detection.
        # They are restored to their original level after the loop completes.
        import logging as _logging
        _smc_logger      = _logging.getLogger('smc_strategy.SMCStrategy')
        _original_level  = _smc_logger.level
        _smc_logger.setLevel(_logging.WARNING)

        try:
            for i in range(window_size, len(m15_df) - forward_bars, step):
                cur_time = m15_df.index[i]

                # Skip Asian session windows - they produce low-quality training samples
                utc_hour = cur_time.hour if hasattr(cur_time, 'hour') else 12
                if utc_hour >= 22 or utc_hour < 7:
                    _cnt_asian += 1
                    continue

                # Build time-bounded context slices
                d1_ctx = d1_df[d1_df.index <= cur_time].tail(50)
                h1_ctx = h1_df[h1_df.index <= cur_time].tail(150)
                m15_win = m15_df.iloc[i - window_size: i].copy()

                if len(d1_ctx) < 20 or len(h1_ctx) < 50 or len(m15_win) < window_size:
                    _cnt_no_ctx += 1
                    continue

                _cnt_total += 1

                # --- Phase 1: HTF trend (D1) via real SMC ---
                try:
                    htf_trend = self.smc.determine_htf_trend(d1_ctx)
                except Exception as _e:
                    self.logger.debug("HTF trend error at window %d: %s", i, _e)
                    continue

                if htf_trend.get('trend') == 'RANGING':
                    _cnt_ranging += 1
                    continue

                direction  = htf_trend['trend']  # 'BULLISH' or 'BEARISH'
                smc_dir    = direction            # alias for clarity

                

                # --- Phase 2: Structure detection (H1) via real SMC ---
                setup_type = None
                poi        = None
                bos_events = []

                try:
                    bos_events = self.smc.detect_break_of_structure(h1_ctx, smc_dir)
                    mss_event  = self.smc.detect_market_structure_shift(
                        h1_ctx, smc_dir, symbol=symbol)

                    if len(bos_events) >= 2:
                        # BOS: Priority 1 = BB, Fallback = OB. Matches live scanner.
                        setup_type = 'BOS'
                        breakers   = self.smc.detect_breaker_blocks(
                            h1_ctx,
                            smc_dir,
                            float(htf_trend.get('swing_high', 0)),
                            float(htf_trend.get('swing_low', 0)),
                        )
                        if breakers:
                            poi = breakers[0]
                        else:
                            obs = self.smc.detect_order_blocks(
                                h1_ctx, smc_dir, symbol=symbol)
                            if obs:
                                poi = obs[0]

                    elif mss_event:
                        # MSS: Priority 1 = OB, Fallback = BB. Matches live scanner.
                        # Previously only OBs were checked here, causing MSS windows
                        # with no OBs to be silently dropped. This meant the training
                        # data never contained MSS+BB setups even though the live
                        # scanner generates them, creating a training/live mismatch.
                        setup_type  = 'MSS'
                        mss_dir     = mss_event.get('direction', smc_dir)
                        obs         = self.smc.detect_order_blocks(
                            h1_ctx, mss_dir, symbol=symbol)
                        if obs:
                            poi = obs[0]
                        else:
                            breakers = self.smc.detect_breaker_blocks(
                                h1_ctx, mss_dir,
                                float(htf_trend.get('swing_high', 0)),
                                float(htf_trend.get('swing_low', 0)),
                            )
                            if breakers:
                                poi = breakers[0]
                except Exception as _e:
                    self.logger.debug("Structure detection error at window %d: %s", i, _e)
                    continue

                if poi is None or setup_type is None:
                    _cnt_no_poi += 1
                    continue

                # Reject mitigated POIs — live scanner never trades them.
                if self.smc.is_poi_mitigated(poi, h1_ctx):
                    _cnt_no_poi += 1
                    continue

                # --- Phase 3: Entry/SL via real SMC ---
                try:
                    atr_val    = self.smc._calculate_atr(m15_win.tail(20))
                    entry_cfg  = self.smc.calculate_entry_price(
                        poi,
                        'UNICORN' if str(poi.get('type', '')).upper() == 'UNICORN' else 'STANDARD',
                        75,
                    )
                    sl_cfg     = self.smc.calculate_stop_loss(
                        poi,
                        'BUY' if direction == 'BULLISH' else 'SELL',
                        symbol,
                        atr_val,
                    )
                    entry      = float(entry_cfg['entry_price'])
                    sl         = float(sl_cfg['stop_loss'])

                    if direction == 'BULLISH' and sl >= entry:
                        _cnt_bad_entry += 1
                        continue
                    if direction == 'BEARISH' and sl <= entry:
                        _cnt_bad_entry += 1
                        continue
                except Exception as _e:
                    self.logger.debug("Entry/SL error at window %d: %s", i, _e)
                    continue

                risk = abs(entry - sl)
                if risk < 1e-9:
                    _cnt_bad_entry += 1
                    continue
                
                # Quality pre-filter: discard low-quality setups before labeling.
                # These setups have ~50% win rate and add noise to training data.
                try:
                    quality_score = self.smc.score_setup_quality(
                        m15_win, poi, htf_trend, bos_events, direction, symbol)
                except Exception:
                    quality_score = 0

                if quality_score < TRAINING_MIN_QUALITY_SCORE:
                    _cnt_low_quality += 1
                    continue

                # --- Phase 4: Feature extraction (exact same fn as live) ---
                try:
                    f = self.extract_features(m15_win, poi, htf_trend, setup_type)
                except Exception as _e:
                    self.logger.debug("Feature extraction error at window %d: %s", i, _e)
                    continue

                # --- Phase 5: Label WIN/LOSS from future bars (chronological) ---
                # forward_bars matches live H1 setup expiry of 8 hours on M15
                # (8 hours x 4 bars per hour = 32 bars).
                # Using 80 bars (20 hours) incorrectly includes price action that
                # occurs after a real order would have expired, inflating WIN rate.
                _expiry_bars = 32  # Matches live H1 setup expiry of 8 hours on M15 (4 bars per hour)
                future = m15_df.iloc[i: i + _expiry_bars]
                if len(future) < 10:
                    continue

                # Use config.MIN_RR_TP2 (3.0) to match the live system exactly.
                # Training with 2.5 while live uses 3.0 means the model is
                # calibrated for different price levels than it actually operates on.
                target_reward = risk * config.MIN_RR_TP2
                is_buy = direction == 'BULLISH'

                # Step 1: Verify the limit order would have filled.
                # In live trading the entry is a limit order placed at the zone
                # boundary. If price never pulls back to that level, the order
                # expires unfilled. Labeling those windows as WIN or LOSS would
                # teach the model that entries execute when they do not.
                entry_filled_at = None
                for _fill_idx, _bar in enumerate(future.itertuples()):
                    if is_buy and float(_bar.low) <= entry:
                        entry_filled_at = _fill_idx
                        break
                    if not is_buy and float(_bar.high) >= entry:
                        entry_filled_at = _fill_idx
                        break

                if entry_filled_at is None:
                    # Price never reached entry zone — limit order expired unfilled.
                    # This would be a no-trade in live. Discard the window.
                    _cnt_unfilled += 1
                    continue

                # Step 2: From the fill bar onward, track TP or SL chronologically.
                # The FIRST level reached determines the label.
                # Checking max/min over the whole window simultaneously corrupts
                # labels when price touches both levels in an unknown sequence.
                label     = None
                _tp_price = (entry + target_reward) if is_buy else (entry - target_reward)
                post_fill = future.iloc[entry_filled_at:]

                for _bar in post_fill.itertuples():
                    if is_buy:
                        if float(_bar.high) >= _tp_price:
                            label = 1.0   # TP reached first
                            break
                        if float(_bar.low) <= sl:
                            label = 0.0   # SL reached first
                            break
                    else:
                        if float(_bar.low) <= _tp_price:
                            label = 1.0   # TP reached first
                            break
                        if float(_bar.high) >= sl:
                            label = 0.0   # SL reached first
                            break

                if label is None:
                    # Neither level reached within the expiry window.
                    # This is an inconclusive outcome — discard rather than guess.
                    _cnt_no_label += 1
                    continue

                _cnt_labeled += 1
                feats.append(f)
                labels.append(label)

        except Exception as e:
            self.logger.error("Sample generation error for %s: %s", symbol, e)

        finally:
            # Always restore the SMC logger level even if an exception occurred
            _smc_logger.setLevel(_original_level)

        self.logger.info(
            "%s sample generation summary: "
            "total=%d  asian_skip=%d  no_ctx=%d  ranging=%d  no_poi=%d  "
            "bad_entry=%d  low_quality=%d  unfilled=%d  inconclusive=%d  labeled=%d",
            symbol,
            _cnt_total, _cnt_asian, _cnt_no_ctx, _cnt_ranging, _cnt_no_poi,
            _cnt_bad_entry, _cnt_low_quality, _cnt_unfilled, _cnt_no_label, _cnt_labeled
        )

        return feats, labels

    # ==================== MODEL TRAINING ====================

    def _train_all_models(self, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            # Chronological split: first 80% = train, last 20% = test.
            # Random split causes look-ahead bias on time-series data.
            split_idx    = int(len(X) * 0.8)
            X_tr, X_te   = X[:split_idx], X[split_idx:]
            y_tr, y_te   = y[:split_idx], y[split_idx:]

            self.logger.info(
                "Chronological split: Train=%d Test=%d. "
                "Train win rate: %.1f%%. Test win rate: %.1f%%.",
                len(X_tr), len(X_te), y_tr.mean() * 100, y_te.mean() * 100)

            self.scaler = StandardScaler()
            X_tr_s = self.scaler.fit_transform(X_tr)
            X_te_s = self.scaler.transform(X_te)

            ok1 = self._fit_xgboost(X_tr_s, y_tr, X_te_s, y_te)
            ok2 = self._fit_lstm_sub(X_tr_s, y_tr, X_te_s, y_te)
            self._fit_random_forest(X_tr_s, y_tr, X_te_s, y_te)

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
                'objective':        'binary:logistic',
                'max_depth':        3,
                'learning_rate':    0.02,
                'subsample':        0.6,
                'colsample_bytree': 0.6,
                'min_child_weight': 30,
                'gamma':            0.5,
                'reg_alpha':        0.5,
                'reg_lambda':       3.0,
                'eval_metric':      'auc',
                'scale_pos_weight': float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1.0),
                'seed':             42,
                }
            self.xgboost_model = xgb.train(
                params, dtrain, num_boost_round=500,
                evals=[(dtest, 'test')], early_stopping_rounds=50,
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

        if acc > 0.85:
            self.logger.warning(
                "XGBoost accuracy %.1f%% is suspiciously high. "
                "This indicates overfitting. Expected range for this strategy: 65-80%%.",
                acc * 100,
            )
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
                n_estimators=300,
                max_depth=2,
                learning_rate=0.02,
                subsample=0.5,
                min_samples_leaf=40,
                min_samples_split=80,
                max_features='sqrt',
                random_state=99)
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
        
        
    def _fit_random_forest(self, X_tr, y_tr, X_te, y_te) -> bool:
        """
        RandomForestClassifier - third ensemble model.
        Provides genuine diversity vs the two boosting models.
        Non-fatal if it fails: ensemble falls back to XGB + GBC.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            self.rf_model = RandomForestClassifier(
                n_estimators=500,
                max_depth=4,
                min_samples_leaf=50,
                min_samples_split=100,
                max_features='sqrt',
                class_weight='balanced',
                n_jobs=-1,
                random_state=77)
            self.rf_model.fit(X_tr, y_tr)
            y_prob = self.rf_model.predict_proba(X_te)[:, 1]
            acc    = accuracy_score(y_te, (y_prob > 0.5).astype(int))
            auc    = roc_auc_score(y_te, y_prob)
            self.logger.info(
                "RandomForest training complete. Accuracy: %.1f%%  AUC: %.3f",
                acc * 100, auc)
            self.training_metadata.update({
                'rf_accuracy': f"{acc * 100:.1f}%",
                'rf_auc':      f"{auc:.3f}",
            })
            return True
        except Exception as e:
            self.logger.error("RandomForest training failed (non-critical): %s", e)
            self.rf_model = None
            return True  # Non-fatal

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
                success = self._train_all_models(X, y)
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
            assert len(f) == _FEATURE_DIM, (
                f"Feature vector length {len(f)} does not match _FEATURE_DIM {_FEATURE_DIM}. "
                "Update _FEATURE_DIM or fix extract_features.")
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

            # Include Random Forest if available. Weights: XGB 50%, LSTM 30%, RF 20%.
            # Falls back to XGB 60% / LSTM 40% if RF model not trained.
            rf_s = None
            if self.models_trained and self.rf_model is not None:
                try:
                    f_s  = self.scaler.transform(features.reshape(1, -1))
                    rf_s = int(self.rf_model.predict_proba(f_s)[0][1] * 100)
                except Exception as rf_err:
                    self.logger.debug("RF prediction skipped: %s", rf_err)

            if rf_s is not None:
                consensus = int(xgb_s * 0.50 + lstm_s * 0.30 + rf_s * 0.20)
            else:
                consensus = int(lstm_s * 0.40 + xgb_s * 0.60)

            # Agreement reflects signal strength (consensus score), not model spread.
            # 75%+ = STRONG  -> auto-execute eligible
            # 60-74% = MODERATE -> auto-execute eligible
            # Below 60% = WEAK -> notify only, never auto-execute
            agreement = (
                'STRONG'    if consensus >= 75 else
                ('MODERATE' if consensus >= 60 else 'WEAK')
            )
            
            self.logger.info(
                "Ensemble: LSTM=%d%%  XGBoost=%d%%  RF=%s%%  Consensus=%d%%  "
                "Agreement=%s  Trained: %s",
                lstm_s, xgb_s, rf_s if rf_s is not None else 'N/A',
                consensus, agreement, self.models_trained)
            return {
                'lstm_score':    lstm_s,
                'xgboost_score': xgb_s,
                'rf_score':      rf_s,
                'consensus_score': consensus,
                'agreement':     agreement,
                'direction':     poi.get('direction', 'BULLISH'),
                'features':      features,
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

    def should_auto_execute(self, consensus_score: int, agreement: str = 'MODERATE') -> bool:
        # WEAK means consensus is below 60%. Notify user but never place a trade.
        # MODERATE (60-74%) and STRONG (75%+) are both auto-execute eligible
        # provided MT5 is connected and consensus meets ML_AUTO_EXECUTE_THRESHOLD.
        if agreement == 'WEAK':
            return False
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
            'rf_accuracy':      self.training_metadata.get('rf_accuracy',      'N/A'),
            'xgboost_auc':      self.training_metadata.get('xgboost_auc',      'N/A'),
            'rf_loaded':        self.rf_model is not None,
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
