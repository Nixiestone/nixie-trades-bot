import logging
import os
import pickle
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import config
import utils

logger = logging.getLogger(__name__)

_MODEL_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
_XGB_PATH    = os.path.join(_MODEL_DIR, 'xgboost_model.pkl')
_SCALER_PATH = os.path.join(_MODEL_DIR, 'scaler.pkl')
_RF_PATH     = os.path.join(_MODEL_DIR, 'rf_model.pkl')
_META_PATH   = os.path.join(_MODEL_DIR, 'training_metadata.pkl')
_PAIR_MODEL_PREFIX = 'pair_xgboost_'
_PAIR_META_PREFIX  = 'pair_metadata_'
_FEATURE_DIM = 28

# Only train on setups scoring >= this value. Weak setups dilute training and
# amplify the class imbalance problem on historical labels.
TRAINING_MIN_QUALITY_SCORE = int(
    getattr(config, 'MIN_SETUP_QUALITY_SCORE', 70))


class MLEnsemble:
    """
    ML ensemble: XGBoost + RandomForest.
    Loads trained models from disk on startup.
    Training uses the real SMCStrategy so training data = live signal data.
    """

    def __init__(self, mt5_connector=None):
        self.logger         = logging.getLogger(f"{__name__}.MLEnsemble")
        self.mt5            = mt5_connector
        self.xgboost_model  = None
        self.rf_model       = None
        self.pair_models: Dict[str, object] = {}
        self.pair_metadata: Dict[str, Dict] = {}
        self.scaler         = StandardScaler()
        self.models_trained = False
        self.training_metadata: Dict = {}

        # Lazy-load SMC strategy to avoid circular import at module level
        self._smc = None

        # Live outcome accumulator for adaptive auto-retraining.
        self.setups_since_training: int  = 0
        self.training_threshold:    int  = int(
            getattr(config, 'ML_AUTO_RETRAIN_OUTCOMES', 25))
        self.training_data_history: List[Tuple[np.ndarray, float, str]] = []
        self.max_history_size:      int  = int(
            getattr(config, 'ML_LIVE_HISTORY_LIMIT', 3000))

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

    @staticmethod
    def _normalise_symbol(symbol: Optional[str]) -> str:
        return str(symbol or 'GLOBAL').upper().replace('/', '').replace('\\', '')

    @staticmethod
    def _pair_model_path(symbol: str) -> str:
        safe = MLEnsemble._normalise_symbol(symbol)
        return os.path.join(_MODEL_DIR, f"{_PAIR_MODEL_PREFIX}{safe}.pkl")

    @staticmethod
    def _pair_meta_path(symbol: str) -> str:
        safe = MLEnsemble._normalise_symbol(symbol)
        return os.path.join(_MODEL_DIR, f"{_PAIR_META_PREFIX}{safe}.pkl")

    def _load_from_disk(self) -> bool:
        try:
            if not all(os.path.exists(p) for p in [_XGB_PATH, _SCALER_PATH]):
                self.logger.info(
                    "No saved models found. Run train_models.py once before going live.")
                return False
            with open(_XGB_PATH,    'rb') as f: self.xgboost_model = pickle.load(f)
            with open(_SCALER_PATH, 'rb') as f: self.scaler        = pickle.load(f)
            if os.path.exists(_RF_PATH):
                with open(_RF_PATH, 'rb') as f:
                    self.rf_model = pickle.load(f)
            else:
                self.rf_model = None
                self.logger.info(
                    "RF model file not found. Ensemble will use XGBoost only.")
            if os.path.exists(_META_PATH):
                with open(_META_PATH, 'rb') as f: self.training_metadata = pickle.load(f)
            model_version = int(self.training_metadata.get('model_version', 0) or 0)
            feature_dim = int(self.training_metadata.get('feature_dim', 0) or 0)
            scaler_dim = int(getattr(self.scaler, 'n_features_in_', 0) or 0)
            expected_version = int(getattr(config, 'ML_MODEL_VERSION', 2))
            if (
                model_version != expected_version
                or feature_dim != _FEATURE_DIM
                or scaler_dim != _FEATURE_DIM
            ):
                self.logger.warning(
                    "Ignoring stale ML artifacts: version=%s feature_dim=%s "
                    "scaler_dim=%s expected version=%s feature_dim=%s.",
                    model_version, feature_dim, scaler_dim,
                    expected_version, _FEATURE_DIM,
                )
                self.xgboost_model = None
                self.rf_model = None
                self.models_trained = False
                return False
            self.models_trained = True
            self._load_pair_models()
            self.logger.info(
                "Trained models loaded. Samples: %s. XGBoost accuracy: %s. "
                "RF loaded: %s.",
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
            with open(_XGB_PATH,    'wb') as f: pickle.dump(self.xgboost_model, f)
            with open(_SCALER_PATH, 'wb') as f: pickle.dump(self.scaler,         f)
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
        per_symbol_samples: Dict[str, List[Tuple[np.ndarray, float]]] = defaultdict(list)

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
                for f, l in zip(feats, labels):
                    per_symbol_samples[self._normalise_symbol(symbol)].append((f, l))

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
                self.training_data_history.append((f, l, 'HISTORICAL'))
            self._train_pair_models_from_samples(per_symbol_samples, force=True)
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
          4. Extract the live 28-element feature vector
          5. Label the managed trade outcome across the live expiry window

        Inconclusive windows (no fill, or neither TP1 nor SL reached after fill)
        are discarded. This keeps the training data aligned with the live setup
        structure while avoiding guessed labels.
        """
        feats:  List[np.ndarray] = []
        labels: List[float]      = []
        window_size  = 100
        forward_bars = int(getattr(config, 'H1_SETUP_EXPIRY_BARS_M15', 48))
        # Keep sampling stride independent from label expiry. A tighter stride
        # increases sample density without changing the live trade definition.
        configured_step = int(getattr(config, 'TRAINING_WINDOW_STEP_M15', 4))
        # Cap at 6 bars (90 minutes on M15) to ensure adequate sample density.
        # A step of 12 produces only one window per 3 hours of market data
        # which is the compounding cause of the low labeled count per symbol.
        step = max(1, min(configured_step, 6))
        # Asian session is NOT skipped during training.
        # The live scanner filters Asian setups at broadcast time.
        # Skipping Asian hours here removes 69% of all M15 windows before
        # any SMC analysis runs, which is the primary cause of the low label count.
        apply_asian_skip = False

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
                if apply_asian_skip and (utc_hour >= 22 or utc_hour < 7):
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
                if int(htf_trend.get('confidence', 0)) < int(
                    getattr(config, 'HIGH_RR_MIN_D1_CONFIDENCE', 60)
                ):
                    _cnt_no_ctx += 1
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
                if getattr(config, 'HIGH_RR_BOS_ONLY', True) and setup_type != 'BOS':
                    _cnt_no_poi += 1
                    continue

                # Reject mitigated POIs — live scanner never trades them.
                if self.smc.is_poi_mitigated(poi, h1_ctx):
                    _cnt_no_poi += 1
                    continue

                h1_poi = dict(poi)
                h1_poi['timeframe'] = 'H1'
                h1_poi['role'] = 'PRIMARY'
                h1_poi['symbol'] = symbol

                # --- Phase 3: Entry/SL via real SMC ---
                try:
                    atr_val    = self.smc._calculate_atr(m15_win.tail(20))
                    entry_cfg  = self.smc.calculate_entry_price(
                        h1_poi,
                        'UNICORN' if str(h1_poi.get('type', '')).upper() == 'UNICORN' else 'STANDARD',
                        75,
                    )
                    sl_cfg     = self.smc.calculate_stop_loss(
                        h1_poi,
                        'BUY' if direction == 'BULLISH' else 'SELL',
                        symbol,
                        atr_val,
                    )
                    m15_refined = self.smc.find_nested_refinement(
                        h1_poi,
                        m15_win.tail(80),
                        symbol=symbol,
                        timeframe='M15',
                        htf_swing_high=float(htf_trend.get('swing_high', 0)),
                        htf_swing_low=float(htf_trend.get('swing_low', 0)),
                    )
                    if m15_refined is not None:
                        _entry_boundary = float(
                            m15_refined.get('high', 0)
                            if direction == 'BULLISH'
                            else m15_refined.get('low', 0)
                        )
                        if _entry_boundary > 0:
                            entry_cfg['entry_price'] = round(_entry_boundary, 5)
                            sl_cfg = self.smc.calculate_stop_loss(
                                m15_refined,
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
                    f = self.extract_features(m15_win, h1_poi, htf_trend, setup_type)
                except Exception as _e:
                    self.logger.debug("Feature extraction error at window %d: %s", i, _e)
                    continue

                # --- Phase 5: Label WIN/LOSS from future bars (chronological) ---
                # forward_bars matches the live H1 setup expiry on M15.
                # Using a longer horizon incorrectly includes price action that
                # occurs after a real order would have expired, inflating WIN rate.
                _expiry_bars = forward_bars
                future = m15_df.iloc[i: i + _expiry_bars]
                if len(future) < 10:
                    continue

                try:
                    tp_cfg = self.smc.calculate_take_profits(
                        entry,
                        sl,
                        direction,
                        float(
                            htf_trend.get(
                                'swing_high' if direction == 'BULLISH' else 'swing_low',
                                0,
                            )
                        ),
                        symbol,
                        m15_data=m15_win,
                    )
                    tp1_price = float(tp_cfg.get('tp1', 0))
                    tp2_price = float(tp_cfg.get('tp2', 0))
                    if tp1_price <= 0 or tp2_price <= 0:
                        _cnt_no_label += 1
                        continue
                except Exception as _e:
                    self.logger.debug("TP calc error at window %d: %s", i, _e)
                    _cnt_no_label += 1
                    continue
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

                # Step 2: From the fill bar onward, track the managed-trade outcome.
                # Live trading takes partial profit at TP1 and activates breakeven,
                # so TP1 reached first is already a positive outcome even if TP2
                # never hits later.
                label     = None
                post_fill = future.iloc[entry_filled_at:]

                for _bar in post_fill.itertuples():
                    if is_buy:
                        if float(_bar.high) >= tp1_price:
                            label = 1.0   # TP1 reached first => managed trade is profitable
                            break
                        if float(_bar.low) <= sl:
                            label = 0.0   # SL reached first
                            break
                    else:
                        if float(_bar.low) <= tp1_price:
                            label = 1.0   # TP1 reached first => managed trade is profitable
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
            "step=%d  expiry=%d  tp1_rr=%.1f  tp2_rr=%.1f  "
            "total=%d  asian_skip=%d  asian_filter=%s  no_ctx=%d  ranging=%d  no_poi=%d  "
            "bad_entry=%d  low_quality=%d  unfilled=%d  inconclusive=%d  labeled=%d",
            symbol,
            step, forward_bars, config.MIN_RR_RATIO, config.MIN_RR_TP2,
            _cnt_total, _cnt_asian, 'on' if apply_asian_skip else 'off',
            _cnt_no_ctx, _cnt_ranging, _cnt_no_poi,
            _cnt_bad_entry, _cnt_low_quality, _cnt_unfilled, _cnt_no_label, _cnt_labeled
        )

        return feats, labels

    # ==================== MODEL TRAINING ====================

    def _train_all_models(self, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            # Wipe stale keys from any previous training run (e.g. lstm_accuracy)
            # so the metadata written to disk only reflects the current ensemble.
            self.training_metadata = {
                'samples': len(X),
                'feature_dim': _FEATURE_DIM,
                'model_version': int(getattr(config, 'ML_MODEL_VERSION', 2)),
            }

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
            self._fit_random_forest(X_tr_s, y_tr, X_te_s, y_te)

            if not ok1:
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
                'scale_pos_weight': float(np.sum(np.round(y_tr) == 0)) / max(float(np.sum(np.round(y_tr) == 1)), 1.0),
                'seed':             42,
                }
            self.xgboost_model = xgb.train(
                params, dtrain, num_boost_round=500,
                evals=[(dtest, 'test')], early_stopping_rounds=50,
                verbose_eval=False)
            y_prob = self.xgboost_model.predict(dtest)
            acc = accuracy_score(y_te, (y_prob > 0.5).astype(int))
            auc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5
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
            'trained_at':       datetime.now(timezone.utc).isoformat(),
        })
        return True
        
        
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
            auc    = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5
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

    # ==================== TRANSFER LEARNING ====================

    def _train_pair_models_from_samples(
        self,
        per_symbol_samples: Dict[str, List[Tuple[np.ndarray, float]]],
        force: bool = False,
    ) -> int:
        """
        Fine-tune one XGBoost booster per symbol from the global booster.

        This is transfer learning for this tabular setup: the global model
        learns the shared strategy edge, then each pair model continues boosting
        from that base on symbol-specific outcomes with conservative rounds.
        """
        if self.xgboost_model is None:
            self.logger.info("Pair fine-tuning skipped: no global XGBoost model.")
            return 0

        promoted = 0
        min_samples = int(getattr(config, 'ML_PAIR_MIN_RETRAIN_SAMPLES', 80))
        min_class_fraction = float(getattr(config, 'ML_PAIR_MIN_CLASS_FRACTION', 0.20))

        for symbol, samples in per_symbol_samples.items():
            symbol = self._normalise_symbol(symbol)
            if symbol in ('', 'GLOBAL', 'HISTORICAL'):
                continue
            if len(samples) < min_samples and not force:
                continue

            X = np.array([s[0] for s in samples], dtype=np.float32)
            y = np.array([s[1] for s in samples], dtype=np.float32)
            if len(X) < min_samples:
                self.logger.info(
                    "%s pair fine-tune skipped: %d samples < %d.",
                    symbol, len(X), min_samples)
                continue
            pos_frac = float(y.mean())
            if pos_frac < min_class_fraction or pos_frac > (1.0 - min_class_fraction):
                self.logger.info(
                    "%s pair fine-tune skipped: class balance %.1f%% wins.",
                    symbol, pos_frac * 100)
                continue

            if self._fine_tune_pair_xgboost(symbol, X, y):
                promoted += 1

        return promoted

    def _fine_tune_pair_xgboost(self, symbol: str, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            import xgboost as xgb

            split_idx = max(int(len(X) * 0.8), 1)
            if len(X) - split_idx < 10:
                self.logger.info(
                    "%s pair fine-tune skipped: validation fold too small.", symbol)
                return False

            X_tr, X_te = X[:split_idx], X[split_idx:]
            y_tr, y_te = y[:split_idx], y[split_idx:]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                self.logger.info(
                    "%s pair fine-tune skipped: train/test fold has one class.", symbol)
                return False

            X_tr_s = self.scaler.transform(X_tr)
            X_te_s = self.scaler.transform(X_te)
            dtrain = xgb.DMatrix(X_tr_s, label=y_tr)
            dtest = xgb.DMatrix(X_te_s, label=y_te)
            params = {
                'objective': 'binary:logistic',
                'max_depth': 2,
                'learning_rate': 0.01,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_weight': 10,
                'gamma': 0.3,
                'reg_alpha': 0.7,
                'reg_lambda': 4.0,
                'eval_metric': 'auc',
                'scale_pos_weight': float(np.sum(np.round(y_tr) == 0)) / max(float(np.sum(np.round(y_tr) == 1)), 1.0),
                'seed': 100 + (abs(hash(symbol)) % 10000),
            }

            candidate = xgb.train(
                params,
                dtrain,
                num_boost_round=120,
                evals=[(dtest, 'test')],
                early_stopping_rounds=20,
                verbose_eval=False,
                xgb_model=self.xgboost_model,
            )
            y_prob = candidate.predict(dtest)
            auc = roc_auc_score(y_te, y_prob)
            acc = accuracy_score(y_te, (y_prob > 0.5).astype(int))
            min_auc = float(getattr(config, 'ML_PAIR_MIN_VALIDATION_AUC', 0.52))
            if auc < min_auc:
                self.logger.info(
                    "%s pair model rejected: AUC %.3f below %.3f.",
                    symbol, auc, min_auc)
                return False

            metadata = {
                'symbol': symbol,
                'samples': int(len(X)),
                'win_rate': float(y.mean()),
                'accuracy': f"{acc * 100:.1f}%",
                'auc': f"{auc:.3f}",
                'feature_dim': _FEATURE_DIM,
                'model_version': int(getattr(config, 'ML_MODEL_VERSION', 3)),
                'trained_at': datetime.now(timezone.utc).isoformat(),
                'base_samples': self.training_metadata.get('samples', 0),
            }
            if self._save_pair_model(symbol, candidate, metadata):
                self.logger.info(
                    "%s pair model promoted: samples=%d AUC=%.3f Acc=%.1f%%.",
                    symbol, len(X), auc, acc * 100)
                return True
            return False

        except ImportError:
            self.logger.warning(
                "Pair transfer learning requires xgboost. Install xgboost.")
            return False
        except Exception as e:
            self.logger.error("%s pair fine-tune failed: %s", symbol, e)
            return False

    # ==================== LIVE AUTO-RETRAIN ====================

    def load_live_training_history(self, limit: Optional[int] = None) -> int:
        """
        Reload persisted live outcomes from the database so auto-training
        survives process restarts.
        """
        try:
            import database as db

            rows = db.get_recent_ml_training_data(
                limit or int(getattr(config, 'ML_LIVE_HISTORY_LIMIT', 3000)))
            loaded: List[Tuple[np.ndarray, float, str]] = []
            # Database returns newest first; reverse to keep chronological order.
            for row in reversed(rows):
                raw_features = json.loads(row.get('features_json') or '[]')
                features = np.array(raw_features, dtype=np.float32)
                if len(features) != _FEATURE_DIM:
                    continue
                outcome = float(row.get('outcome'))
                if outcome not in (0.0, 1.0):
                    continue
                symbol = self._normalise_symbol(row.get('symbol'))
                loaded.append((features, outcome, symbol))

            if loaded:
                self.training_data_history = loaded[-self.max_history_size:]
                self.logger.info(
                    "Loaded %d live ML training examples from database.",
                    len(self.training_data_history))
            return len(self.training_data_history)
        except Exception as e:
            self.logger.warning("Could not load live ML training history: %s", e)
            return 0

    def retrain_from_live_history(self, force: bool = False) -> bool:
        """Train from durable live outcomes when enough examples exist."""
        if len(self.training_data_history) < int(
            getattr(config, 'ML_MIN_RETRAIN_SAMPLES', 200)
        ):
            if force:
                self.logger.warning(
                    "Live retrain skipped: only %d samples available.",
                    len(self.training_data_history))
            return False

    def _load_pair_models(self) -> int:
        loaded = 0
        try:
            self.pair_models = {}
            self.pair_metadata = {}
            expected_version = int(getattr(config, 'ML_MODEL_VERSION', 3))

            for filename in os.listdir(_MODEL_DIR):
                if not filename.startswith(_PAIR_MODEL_PREFIX) or not filename.endswith('.pkl'):
                    continue
                symbol = filename[len(_PAIR_MODEL_PREFIX):-4].upper()
                model_path = os.path.join(_MODEL_DIR, filename)
                meta_path = self._pair_meta_path(symbol)
                metadata = {}
                if os.path.exists(meta_path):
                    with open(meta_path, 'rb') as f:
                        metadata = pickle.load(f)
                if (
                    int(metadata.get('model_version', 0) or 0) != expected_version
                    or int(metadata.get('feature_dim', 0) or 0) != _FEATURE_DIM
                ):
                    self.logger.warning(
                        "Skipping stale pair model for %s.", symbol)
                    continue
                with open(model_path, 'rb') as f:
                    self.pair_models[symbol] = pickle.load(f)
                self.pair_metadata[symbol] = metadata
                loaded += 1
            if loaded:
                self.logger.info("Loaded %d pair fine-tuned model(s).", loaded)
        except Exception as e:
            self.logger.warning("Could not load pair models: %s", e)
        return loaded

    def _save_pair_model(self, symbol: str, model: object, metadata: Dict) -> bool:
        try:
            symbol = self._normalise_symbol(symbol)
            with open(self._pair_model_path(symbol), 'wb') as f:
                pickle.dump(model, f)
            with open(self._pair_meta_path(symbol), 'wb') as f:
                pickle.dump(metadata, f)
            self.pair_models[symbol] = model
            self.pair_metadata[symbol] = metadata
            return True
        except Exception as e:
            self.logger.error("Failed to save pair model for %s: %s", symbol, e)
            return False

        X = np.array([h[0] for h in self.training_data_history], dtype=np.float32)
        y = np.array([h[1] for h in self.training_data_history], dtype=np.float32)
        if len(np.unique(y)) < 2:
            self.logger.warning(
                "Live retrain skipped: outcomes contain only one class.")
            return False
        if not self._train_all_models(X, y):
            return False

        per_symbol_samples: Dict[str, List[Tuple[np.ndarray, float]]] = defaultdict(list)
        for features, outcome, symbol in self.training_data_history:
            per_symbol_samples[self._normalise_symbol(symbol)].append((features, outcome))
        promoted = self._train_pair_models_from_samples(per_symbol_samples, force=force)
        self.logger.info("Live transfer-learning pass promoted %d pair model(s).", promoted)
        return True

    def record_trade_outcome(
        self,
        features: np.ndarray,
        won: bool,
        symbol: Optional[str] = None,
    ) -> bool:
        """
        Record a live trade outcome. Called by position_monitor after each close.
        Retrains automatically after the configured number of new outcomes.

        Args:
            features: 28-element vector from get_ensemble_prediction()['features']
            won:      True = profitable managed outcome. False = loss.

        Returns:
            True if a retrain was triggered and succeeded.
        """
        features = np.array(features, dtype=np.float32)
        if len(features) != _FEATURE_DIM:
            self.logger.warning(
                "Ignoring ML outcome with feature length %d; expected %d.",
                len(features), _FEATURE_DIM)
            return False

        self.training_data_history.append(
            (features, 1.0 if won else 0.0, self._normalise_symbol(symbol)))
        if len(self.training_data_history) > self.max_history_size:
            self.training_data_history = self.training_data_history[-self.max_history_size:]

        self.setups_since_training += 1

        if self.setups_since_training >= self.training_threshold:
            self.logger.info(
                "Auto-retrain triggered: %d live outcomes accumulated.",
                self.training_threshold)
            self.setups_since_training = 0
            success = self.retrain_from_live_history(force=True)
            if success:
                self.logger.info(
                    "Auto-retrain complete. Total samples: %d.",
                    len(self.training_data_history))
            return success
        return False

    # ==================== FEATURE EXTRACTION ====================

    def extract_features(
        self, data: pd.DataFrame, poi: Dict, htf_trend: Dict, setup_type: str
    ) -> np.ndarray:
        """
        Convert OHLCV data and POI into a 28-element normalised feature vector.

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
         22    Gold instrument flag
         23    Crypto instrument flag
         24    JPY pair/cross flag
         25    London/NY/overlap session flag
         26    Current candle range / ATR
         27    20-bar directional efficiency
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
            symbol    = str(poi.get('symbol', '')).upper()
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

            # 9: Freshness — POI index is relative to the data window passed in.
            # Clamp to [0, len(data)-1] to guard against cross-timeframe index values.
            _poi_idx = int(poi.get('index', len(data) - 1))
            _poi_idx = max(0, min(_poi_idx, len(data) - 1))
            bars_ago = max(0, (len(data) - 1) - _poi_idx)
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

            is_gold = 1.0 if symbol == 'XAUUSD' else 0.0
            is_crypto = 1.0 if symbol in ('BTCUSD', 'BTCUSDT') else 0.0
            is_jpy = 1.0 if 'JPY' in symbol else 0.0
            try:
                last_ts = data.index[-1]
                hour = int(getattr(last_ts, 'hour', 0))
            except Exception:
                hour = 0
            liquid_session = 1.0 if 7 <= hour <= 20 else 0.0
            current_range = max(float(high[-1]) - float(low[-1]), 0.0)
            range_atr = min(current_range / max(atr, 1e-9), 5.0) / 5.0
            if len(close) >= 21:
                net_move = abs(float(close[-1]) - float(close[-21]))
                path = float(np.sum(np.abs(np.diff(close[-21:]))))
                efficiency = min(net_move / max(path, 1e-9), 1.0)
            else:
                efficiency = 0.0

            f = np.array([
                htf_ok, rsi / 100.0, rsi_zone, atr_norm, atr_r,
                macd_ok, poi_vr, vol_sg, imp_p, fresh,
                bb_up_r, bb_lo_r, ab50, ab20,
                is_ob, is_bb, is_fvg, is_bos, body_r, cpoi,
                ema_ok, cnt / 10.0,
                is_gold, is_crypto, is_jpy, liquid_session, range_atr, efficiency,
            ], dtype=np.float32)
            assert len(f) == _FEATURE_DIM, (
                f"Feature vector length {len(f)} does not match _FEATURE_DIM {_FEATURE_DIM}. "
                "Update _FEATURE_DIM or fix extract_features.")
            return np.nan_to_num(f, nan=0.0, posinf=1.0, neginf=0.0)

        except Exception as e:
            self.logger.error("Feature extraction error: %s", e)
            return np.zeros(_FEATURE_DIM, dtype=np.float32)

    # ==================== LIVE PREDICTIONS ====================


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

    def predict_pair_xgboost(self, features: np.ndarray, symbol: str) -> Optional[int]:
        """Predict with the symbol-specific fine-tuned booster when available."""
        symbol = self._normalise_symbol(symbol)
        model = self.pair_models.get(symbol)
        if model is None:
            return None
        try:
            import xgboost as xgb
            f_s = self.scaler.transform(features.reshape(1, -1))
            p = model.predict(xgb.DMatrix(f_s))[0]
            return int(p * 100)
        except Exception as e:
            self.logger.debug("%s pair prediction skipped: %s", symbol, e)
            return None

    def get_ensemble_prediction(self, data, poi, htf_trend, setup_type) -> Dict:
        """
        Combined prediction using XGBoost (70%) and RandomForest (30%).
        Falls back to XGBoost alone when the RF model is not trained.

        The 'features' key must be saved and passed to record_trade_outcome()
        when the trade closes, so the models learn from this specific trade.
        """
        try:
            features = self.extract_features(data, poi, htf_trend, setup_type)
            xgb_s    = self.predict_xgboost(data, poi, htf_trend, setup_type)
            symbol   = self._normalise_symbol(poi.get('symbol'))
            pair_s   = self.predict_pair_xgboost(features, symbol)

            rf_s = None
            if self.models_trained and self.rf_model is not None:
                try:
                    f_s  = self.scaler.transform(features.reshape(1, -1))
                    rf_s = int(self.rf_model.predict_proba(f_s)[0][1] * 100)
                except Exception as rf_err:
                    self.logger.debug("RF prediction skipped: %s", rf_err)

            base_consensus = (
                int(xgb_s * 0.70 + rf_s * 0.30)
                if rf_s is not None
                else xgb_s
            )
            if pair_s is not None:
                pair_weight = float(getattr(config, 'ML_PAIR_MODEL_WEIGHT', 0.45))
                consensus = int(
                    base_consensus * (1.0 - pair_weight)
                    + pair_s * pair_weight
                )
            else:
                consensus = base_consensus

            # 75%+ = STRONG  -> auto-execute eligible
            # 60-74% = MODERATE -> auto-execute eligible
            # Below 60% = WEAK -> notify only, never auto-execute
            agreement = (
                'STRONG'    if consensus >= 75 else
                ('MODERATE' if consensus >= 60 else 'WEAK')
            )

            self.logger.info(
                "Ensemble: XGBoost=%d%%  RF=%s%%  Pair=%s%%  Consensus=%d%%  "
                "Agreement=%s  Trained: %s",
                xgb_s,
                rf_s if rf_s is not None else 'N/A',
                pair_s if pair_s is not None else 'N/A',
                consensus, agreement, self.models_trained)

            return {
                'xgboost_score':   xgb_s,
                'rf_score':        rf_s,
                'pair_score':      pair_s,
                'consensus_score': consensus,
                'agreement':       agreement,
                'direction':       poi.get('direction', 'BULLISH'),
                'features':        features,
            }

        except Exception as e:
            self.logger.error("Ensemble prediction error: %s", e)
            return {
                'xgboost_score': 50,
                'rf_score':      None,
                'pair_score':    None,
                'consensus_score': 50,
                'agreement': 'WEAK',
                'direction': poi.get('direction', 'BULLISH'),
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
                    'Run: python train_models.py'),
                'pair_models': len(self.pair_models)}
        return {
            'trained':          True,
            'status_text':      'Trained ML active (global XGBoost/RF plus pair transfer models).',
            'xgboost_accuracy': self.training_metadata.get('xgboost_accuracy', 'N/A'),
            'rf_accuracy':      self.training_metadata.get('rf_accuracy',      'N/A'),
            'xgboost_auc':      self.training_metadata.get('xgboost_auc',      'N/A'),
            'rf_loaded':        self.rf_model is not None,
            'training_date':    self.training_metadata.get('trained_at',       'Unknown'),
            'samples':          self.training_metadata.get('samples',           0),
            'live_outcomes_since_last_retrain': self.setups_since_training,
            'retrain_threshold': self.training_threshold,
            'feature_dim':       self.training_metadata.get('feature_dim', _FEATURE_DIM),
            'model_version':     self.training_metadata.get(
                'model_version', getattr(config, 'ML_MODEL_VERSION', 2)),
            'pair_models':       sorted(self.pair_models.keys()),
        }

    # ==================== HEURISTIC FALLBACKS (PRE-TRAINING) ====================

    

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
        return max(38, min(score, 59))

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
