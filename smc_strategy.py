import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

try:
    from smartmoneyconcepts import smc as _smc_pkg
    _SMC_PKG_AVAILABLE = True
except ImportError:
    _smc_pkg = None
    _SMC_PKG_AVAILABLE = False

import config
import utils

logger = logging.getLogger(__name__)


class SMCStrategy:
    """
    Smart Money Concepts trading strategy with institutional-grade refinements.
    Implements multi-timeframe analysis, Order Block/Breaker detection,
    and 8 precision filters for sniper entries.
    """
    
    def __init__(self):
        """
        Initialize SMC Strategy Engine.
        Uses the smartmoneyconcepts package for swing, OB, FVG, BOS and CHOCH
        detection when available. Falls back to the built-in ATR/volume
        algorithms on all methods if the package is not installed.
        """
        self.logger = logging.getLogger(f"{__name__}.SMCStrategy")
        if _SMC_PKG_AVAILABLE:
            self.logger.info(
                "SMC Strategy Engine initialized with smartmoneyconcepts package.")
        else:
            self.logger.warning(
                "SMC Strategy Engine initialized WITHOUT smartmoneyconcepts package. "
                "Install it: pip install smartmoneyconcepts --break-system-packages")
    
    # ==================== PHASE 1: HTF CONTEXT ====================
    
    def determine_htf_trend(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Determine Higher Timeframe trend direction.
        
        Args:
            data: OHLCV DataFrame for Daily or 4H timeframe
            
        Returns:
            dict: Trend information
        """
        try:
            # Identify swing highs and lows (last 20 swings)
            swings = self._identify_swings(data.tail(100))
            
            if not swings or len(swings) < 5:
                return {
                    'trend': 'RANGING',
                    'confidence': 0,
                    'reason': 'Insufficient swing data'
                }
            
            # Use only the most recent 12 swings for trend determination.
            # Older swings from a prior regime dilute the current trend signal
            # and cause bearish/bullish markets to appear ranging.
            recent = swings[-12:] if len(swings) > 12 else swings
            total  = len(recent)

            hh_count = sum(1 for s in recent if s['type'] == 'HH')
            ll_count = sum(1 for s in recent if s['type'] == 'LL')
            hl_count = sum(1 for s in recent if s['type'] == 'HL')
            lh_count = sum(1 for s in recent if s['type'] == 'LH')

            bullish_ratio = (hh_count + hl_count) / total
            bearish_ratio = (ll_count + lh_count) / total

            # Secondary price confirmation: if the 50-bar SMA direction agrees
            # with the swing classification, lower the threshold from 0.60 to 0.55.
            # This prevents borderline cases being discarded as RANGING.
            try:
                ma50        = float(data['close'].rolling(50).mean().iloc[-1])
                price_now   = float(data['close'].iloc[-1])
                ma_bullish  = price_now > ma50
                threshold   = 0.55 if (
                    (bullish_ratio > bearish_ratio and ma_bullish) or
                    (bearish_ratio > bullish_ratio and not ma_bullish)
                ) else 0.60
            except Exception:
                threshold = 0.60
            
            if bullish_ratio >= threshold:
                # swing_high must come from HIGH direction swings only: HH or LH.
                # HL is a Higher Low — it is a LOW price, not a HIGH price.
                # Using HL here was putting a trough price into the TP target for BUY.
                high_swings = [s for s in swings if s['direction'] == 'HIGH']
                low_swings  = [s for s in swings if s['direction'] == 'LOW']
                _swing_high = (
                    high_swings[-1]['price']
                    if high_swings
                    else max(s['price'] for s in swings)
                )
                _swing_low = (
                    min(s['price'] for s in low_swings)
                    if low_swings
                    else min(s['price'] for s in swings)
                )
                return {
                    'trend':      'BULLISH',
                    'confidence': int(bullish_ratio * 100),
                    'reason':     f'{hh_count} Higher Highs confirmed',
                    'swing_high': _swing_high,
                    'swing_low':  _swing_low,
                }
            elif bearish_ratio >= threshold:
                # swing_low must come from LOW direction swings only: LL or HL.
                # LH is a Lower High — it is a HIGH price, not a LOW price.
                # Using LH here was putting a peak price into the TP target for SELL.
                high_swings = [s for s in swings if s['direction'] == 'HIGH']
                low_swings  = [s for s in swings if s['direction'] == 'LOW']
                _swing_high = (
                    max(s['price'] for s in high_swings)
                    if high_swings
                    else max(s['price'] for s in swings)
                )
                _swing_low = (
                    low_swings[-1]['price']
                    if low_swings
                    else min(s['price'] for s in swings)
                )
                return {
                    'trend':      'BEARISH',
                    'confidence': int(bearish_ratio * 100),
                    'reason':     f'{ll_count} Lower Lows confirmed',
                    'swing_high': _swing_high,
                    'swing_low':  _swing_low,
                }
            else:
                return {
                    'trend': 'RANGING',
                    'confidence': 50,
                    'reason': 'No clear directional bias',
                    'swing_high': max(s['price'] for s in swings),
                    'swing_low': min(s['price'] for s in swings)
                }
        
        except Exception as e:
            self.logger.error(f"Error determining HTF trend: {e}")
            return {
                'trend': 'RANGING',
                'confidence': 0,
                'reason': f'Analysis error: {str(e)}'
            }
    
    def _identify_swings(self, data: pd.DataFrame, lookback: int = 2) -> List[Dict]:
        """
        Identify swing highs and lows using the smartmoneyconcepts package when
        available, falling back to the ATR-based algorithm when the package is
        not installed or returns insufficient data.

        lookback=2 for HTF trend detection (more swings on Daily data).
        lookback=3 for entry-level structure (stricter, reduces noise on M15/H1).

        Output format is identical regardless of which engine is used:
            [{'index': int, 'price': float, 'direction': 'HIGH'|'LOW',
              'type': 'HH'|'HL'|'LH'|'LL'}, ...]
        """
        if _SMC_PKG_AVAILABLE and len(data) >= lookback * 2 + 5:
            try:
                swing_df = _smc_pkg.swing_highs_lows(
                    data, swing_highs_lows=lookback
                )
                swings: List[Dict] = []
                last_high_price: Optional[float] = None
                last_low_price:  Optional[float] = None

                for pos, (row_idx, row) in enumerate(swing_df.iterrows()):
                    hl_val = row.get('HighLow')
                    if pd.isna(hl_val):
                        continue
                    level = row.get('Level')
                    if pd.isna(level):
                        continue

                    hl_int = int(hl_val)
                    price  = float(level)
                    if price <= 0:
                        continue

                    if hl_int == 1:
                        swing_type = (
                            'HH' if (last_high_price is None or price > last_high_price)
                            else 'LH'
                        )
                        last_high_price = price
                        swings.append({
                            'index':     pos,
                            'price':     price,
                            'direction': 'HIGH',
                            'type':      swing_type,
                        })
                    elif hl_int == -1:
                        swing_type = (
                            'LL' if (last_low_price is None or price < last_low_price)
                            else 'HL'
                        )
                        last_low_price = price
                        swings.append({
                            'index':     pos,
                            'price':     price,
                            'direction': 'LOW',
                            'type':      swing_type,
                        })

                if swings:
                    return swings
            except Exception as e:
                self.logger.debug(
                    "SMC package swing detection failed, using fallback: %s", e)

        # Fallback: ATR-filtered custom algorithm.
        swings = []
        highs  = data['high'].values
        lows   = data['low'].values
        close  = data['close'].values
        n      = len(data)

        tr_values = []
        for k in range(1, n):
            tr = max(
                highs[k] - lows[k],
                abs(highs[k] - close[k - 1]),
                abs(lows[k]  - close[k - 1]),
            )
            tr_values.append(tr)
        atr_approx     = float(np.mean(tr_values[-20:])) if len(tr_values) >= 20 else 0.0
        min_swing_size = atr_approx * 0.3

        for i in range(lookback, n - lookback):
            if all(highs[i] > highs[i - j] for j in range(1, lookback + 1)) and \
               all(highs[i] > highs[i + j] for j in range(1, lookback + 1)):
                swing_size = highs[i] - min(lows[max(0, i - lookback): i + lookback + 1])
                if swing_size >= min_swing_size:
                    swings.append({
                        'index': i, 'price': highs[i],
                        'direction': 'HIGH', 'type': None,
                    })

        for i in range(lookback, n - lookback):
            if all(lows[i] < lows[i - j] for j in range(1, lookback + 1)) and \
               all(lows[i] < lows[i + j] for j in range(1, lookback + 1)):
                swing_size = max(highs[max(0, i - lookback): i + lookback + 1]) - lows[i]
                if swing_size >= min_swing_size:
                    swings.append({
                        'index': i, 'price': lows[i],
                        'direction': 'LOW', 'type': None,
                    })

        swings.sort(key=lambda x: x['index'])

        last_high_price = None
        last_low_price  = None

        for swing in swings:
            if swing['direction'] == 'HIGH':
                swing['type'] = (
                    'HH' if (last_high_price is None or swing['price'] > last_high_price)
                    else 'LH'
                )
                last_high_price = swing['price']
            else:
                swing['type'] = (
                    'LL' if (last_low_price is None or swing['price'] < last_low_price)
                    else 'HL'
                )
                last_low_price = swing['price']

        return [s for s in swings if s['type'] is not None]
    
    def detect_order_blocks(
        self,
        data: pd.DataFrame,
        direction: str,
        min_impulse_pips: float = 20,
        symbol: str = 'EURUSD',
    ) -> List[Dict]:
        """
        Detect Order Blocks using the smartmoneyconcepts package.
        Falls back to the ATR/volume custom algorithm when the package is
        unavailable or returns no results.

        Args:
            data:             OHLCV DataFrame
            direction:        'BULLISH' or 'BEARISH'
            min_impulse_pips: Minimum impulse size in pips (fallback only)
            symbol:           Trading symbol for pip size lookup

        Returns:
            list: Valid Order Blocks sorted by confidence descending
        """
        order_blocks = []
        _ob_pip_size = utils.get_pip_value(symbol)
        if _ob_pip_size <= 0:
            _ob_pip_size = 0.0001
        is_bull = direction == 'BULLISH'

        if _SMC_PKG_AVAILABLE and len(data) >= 30:
            try:
                swing_df = _smc_pkg.swing_highs_lows(data, swing_highs_lows=3)
                ob_df    = _smc_pkg.ob(data, swing_df)

                rolling_avg_vol = data['volume'].rolling(20, min_periods=5).mean()

                for i, (idx, row) in enumerate(ob_df.iterrows()):
                    ob_val = row.get('OB')
                    if pd.isna(ob_val):
                        continue

                    # Package: 1 = bullish OB, -1 = bearish OB
                    ob_int = int(ob_val)
                    if is_bull and ob_int != 1:
                        continue
                    if not is_bull and ob_int != -1:
                        continue

                    top    = float(row.get('Top',    0) or 0)
                    bottom = float(row.get('Bottom', 0) or 0)
                    if top <= bottom or top <= 0:
                        continue

                    # Skip mitigated OBs (package fills MitigatedIndex when price
                    # has returned to and closed through the zone).
                    mitigated_idx = row.get('MitigatedIndex')
                    if mitigated_idx is not None and not pd.isna(mitigated_idx):
                        continue

                    # Use get_loc which handles timezone-aware vs naive mismatches.
                    # list.index() raises TypeError on tz-aware timestamps causing
                    # silent fallback to the fallback algorithm with strict filtering.
                    try:
                        raw_pos = data.index.get_loc(idx)
                        if isinstance(raw_pos, slice):
                            pos = raw_pos.start
                        elif isinstance(raw_pos, np.ndarray):
                            pos = int(np.where(raw_pos)[0][0])
                        else:
                            pos = int(raw_pos)
                    except Exception:
                        try:
                            ts_arr = pd.to_datetime(data.index)
                            ts_val = pd.to_datetime(idx)
                            pos    = int(np.argmin(np.abs(ts_arr - ts_val)))
                        except Exception:
                            pos = i

                    # Volume confirmation
                    avg_vol = float(rolling_avg_vol.iloc[pos]) if pos < len(rolling_avg_vol) else 1.0
                    if avg_vol != avg_vol or avg_vol <= 0:
                        avg_vol = 1.0
                    candle      = data.iloc[pos]
                    candle_vol  = float(candle.get('volume', 1.0))
                    vol_ratio   = candle_vol / avg_vol if avg_vol > 0 else 1.0
                    if vol_ratio < config.VOLUME_MULTIPLIER_OB:
                        continue

                    ob_volume = float(row.get('OBVolume', 0) or 0)
                    impulse_pips_val = (
                        abs(top - bottom) / _ob_pip_size
                        if _ob_pip_size > 0 else 0.0
                    )
                    percentage = float(row.get('Percentage', 0) or 0)

                    order_blocks.append({
                        'type':         'OB',
                        'direction':    direction,
                        'index':        pos,
                        'timestamp':    idx,
                        'high':         top,
                        'low':          bottom,
                        'open':         float(candle.get('open',  bottom)),
                        'close':        float(candle.get('close', top)),
                        'volume_ratio': round(vol_ratio, 3),
                        'impulse_pips': round(impulse_pips_val, 1),
                        'confidence':   self._calculate_ob_confidence(
                            vol_ratio,
                            percentage,
                            impulse_pips_val,
                        ),
                    })

                if order_blocks:
                    order_blocks.sort(key=lambda x: x['confidence'], reverse=True)
                    self.logger.info(
                        "SMC package detected %d Order Blocks for %s.",
                        len(order_blocks), direction)
                    return order_blocks
            except Exception as e:
                self.logger.debug(
                    "SMC package OB detection failed, using fallback: %s", e)

        # Fallback: ATR/volume custom algorithm.
        try:
            rolling_avg_vol = data['volume'].rolling(20, min_periods=5).mean()

            for i in range(10, len(data) - 6):
                candle       = data.iloc[i]
                next_candles = data.iloc[i + 1:i + 6]

                avg_volume = float(rolling_avg_vol.iloc[i])
                if avg_volume != avg_volume or avg_volume <= 0:
                    avg_volume = max(float(candle['volume']), 1e-9)

                is_bullish_candle = candle['close'] > candle['open']
                is_bearish_candle = candle['close'] < candle['open']

                if direction == 'BULLISH' and not is_bearish_candle:
                    continue
                if direction == 'BEARISH' and not is_bullish_candle:
                    continue

                if direction == 'BULLISH':
                    impulse_move = next_candles['high'].max() - candle['low']
                else:
                    impulse_move = candle['high'] - next_candles['low'].min()

                impulse_pips = impulse_move / _ob_pip_size

                _recent_tr   = data['high'].iloc[max(0, i - 14):i] - data['low'].iloc[max(0, i - 14):i]
                _atr_price   = float(_recent_tr.mean()) if len(_recent_tr) > 0 else 0.0
                _atr_pips_ob = (_atr_price / _ob_pip_size) if _ob_pip_size > 0 else 20.0
                _min_impulse = max(
                    config.OB_IMPULSE_MIN_PIPS_FLOOR,
                    _atr_pips_ob * config.OB_IMPULSE_ATR_MULTIPLIER,
                )
                if impulse_pips < _min_impulse:
                    continue

                _candle_vol  = float(candle['volume'])
                _impulse_vol = float(next_candles['volume'].mean()) if len(next_candles) > 0 else 0.0
                volume_ratio   = _candle_vol  / avg_volume
                impulse_volume = _impulse_vol / avg_volume

                if not (volume_ratio == volume_ratio) or volume_ratio <= 0:
                    continue
                if not (impulse_volume == impulse_volume) or impulse_volume <= 0:
                    continue
                if volume_ratio < config.VOLUME_MULTIPLIER_OB:
                    continue
                if impulse_volume < config.VOLUME_MULTIPLIER_IMPULSE:
                    continue

                candle_body_ratio = (
                    abs(candle['close'] - candle['open'])
                    / max(candle['high'] - candle['low'], 1e-9)
                )
                if candle_body_ratio < 0.4:
                    continue

                order_blocks.append({
                    'type':         'OB',
                    'direction':    direction,
                    'index':        i,
                    'timestamp':    data.index[i],
                    'high':         float(candle['high']),
                    'low':          float(candle['low']),
                    'open':         float(candle['open']),
                    'close':        float(candle['close']),
                    'volume_ratio': round(volume_ratio, 3),
                    'impulse_pips': round(impulse_pips, 1),
                    'confidence':   self._calculate_ob_confidence(
                        volume_ratio, impulse_volume, impulse_pips),
                })

            order_blocks.sort(key=lambda x: x['confidence'], reverse=True)
            self.logger.info(
                "Fallback detected %d Order Blocks for %s.", len(order_blocks), direction)
            return order_blocks

        except Exception as e:
            self.logger.error("Error detecting Order Blocks: %s", e)
            return []
    
    def detect_breaker_blocks(
        self,
        data: pd.DataFrame,
        direction: str,
        htf_swing_high: float,
        htf_swing_low: float,
        symbol: str = 'EURUSD',
    ) -> List[Dict]:
        """
        Detect Breaker Blocks (failed support/resistance zones).
        
        Args:
            data: OHLCV DataFrame
            direction: 'BULLISH' or 'BEARISH'
            htf_swing_high: HTF swing high level
            htf_swing_low: HTF swing low level
            
        Returns:
            list: Valid Breaker Blocks
        """
        breakers = []
        
        try:
            rolling_avg_vol = data['volume'].rolling(20, min_periods=5).mean()

            for i in range(20, len(data) - 5):
                candle       = data.iloc[i]
                prev_candles = data.iloc[i-10:i]
                next_candles = data.iloc[i+1:i+5]

                # Minimum volume confirmation for Breaker Blocks.
                # A BB on near-zero volume is not institutional activity.
                _avg_vol = float(rolling_avg_vol.iloc[i])
                if _avg_vol <= 0 or _avg_vol != _avg_vol:
                    _avg_vol = max(float(candle['volume']), 1e-9)
                _candle_vol_ratio = float(candle['volume']) / _avg_vol
                if not (_candle_vol_ratio == _candle_vol_ratio) or _candle_vol_ratio < 1.2:
                    continue

                if direction == 'BULLISH':
                    # Look for broken resistance becoming support
                    resistance = prev_candles['high'].max()
                    
                    # Check if price broke above resistance
                    if candle['close'] > resistance:
                        # Check if price held above on pullback
                        pullback_low = next_candles['low'].min()
                        
                        if pullback_low >= candle['low'] * 0.995:  # Within 0.5%
                            _bb_pip_sz = utils.get_pip_value(symbol)
                            if _bb_pip_sz <= 0:
                                _bb_pip_sz = 0.0001
                            _bb_impulse = max((candle['close'] - resistance) / _bb_pip_sz, 0.0)
                            breakers.append({
                                'type': 'BB',
                                'direction': direction,
                                'index': i,
                                'timestamp': data.index[i],
                                'high': candle['high'],
                                'low': candle['low'],
                                'breaker_level': resistance,
                                'volume_ratio': _candle_vol_ratio,
                                'impulse_pips': _bb_impulse,
                                'confidence': 75
                            })
                
                else:  # BEARISH
                    # Look for broken support becoming resistance
                    support = prev_candles['low'].min()
                    
                    # Check if price broke below support
                    if candle['close'] < support:
                        # Check if price held below on pullback
                        pullback_high = next_candles['high'].max()
                        
                        if pullback_high <= candle['high'] * 1.005:
                            _bb_pip_sz = utils.get_pip_value(symbol)
                            if _bb_pip_sz <= 0:
                                _bb_pip_sz = 0.0001
                            _bb_impulse = max((support - candle['close']) / _bb_pip_sz, 0.0)
                            breakers.append({
                                'type': 'BB',
                                'direction': direction,
                                'index': i,
                                'timestamp': data.index[i],
                                'high': candle['high'],
                                'low': candle['low'],
                                'breaker_level': support,
                                'volume_ratio': _candle_vol_ratio,
                                'impulse_pips': _bb_impulse,
                                'confidence': 75
                            })
            
            self.logger.info(f"Detected {len(breakers)} valid Breaker Blocks for {direction} direction")
            return breakers
        
        except Exception as e:
            self.logger.error(f"Error detecting Breaker Blocks: {e}")
            return []
    
    def detect_fair_value_gaps(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect Fair Value Gaps (3-candle price imbalances) using the
        smartmoneyconcepts package. Falls back to the 3-candle custom
        algorithm when the package is unavailable.

        Args:
            data: OHLCV DataFrame

        Returns:
            list: FVG dicts with keys type, direction, index, timestamp,
                  high, low, gap_size, filled
        """
        fvgs = []

        if _SMC_PKG_AVAILABLE and len(data) >= 3:
            try:
                fvg_df = _smc_pkg.fvg(data, join_consecutive=False)
                index_list = list(data.index)

                for i, (idx, row) in enumerate(fvg_df.iterrows()):
                    fvg_val = row.get('FVG')
                    if pd.isna(fvg_val):
                        continue

                    top    = float(row.get('Top',    0) or 0)
                    bottom = float(row.get('Bottom', 0) or 0)
                    if top <= bottom or top <= 0:
                        continue

                    fvg_int   = int(fvg_val)
                    direction = 'BULLISH' if fvg_int == 1 else 'BEARISH'

                    try:
                        raw_pos = data.index.get_loc(idx)
                        if isinstance(raw_pos, slice):
                            pos = raw_pos.start
                        elif isinstance(raw_pos, np.ndarray):
                            pos = int(np.where(raw_pos)[0][0])
                        else:
                            pos = int(raw_pos)
                    except Exception:
                        pos = i

                    mitigated = row.get('MitigatedIndex')
                    is_filled = (
                        mitigated is not None and not pd.isna(mitigated)
                    )

                    fvgs.append({
                        'type':      'FVG',
                        'direction': direction,
                        'index':     pos,
                        'timestamp': idx,
                        'high':      top,
                        'low':       bottom,
                        'gap_size':  round(top - bottom, 8),
                        'filled':    is_filled,
                    })

                self.logger.info(
                    "SMC package detected %d Fair Value Gaps.", len(fvgs))
                return fvgs
            except Exception as e:
                self.logger.debug(
                    "SMC package FVG detection failed, using fallback: %s", e)

        # Fallback: 3-candle imbalance algorithm.
        try:
            for i in range(2, len(data)):
                candle_1 = data.iloc[i - 2]
                candle_3 = data.iloc[i]

                if candle_1['high'] < candle_3['low']:
                    gap_size = candle_3['low'] - candle_1['high']
                    fvgs.append({
                        'type':      'FVG',
                        'direction': 'BULLISH',
                        'index':     i - 1,
                        'timestamp': data.index[i - 1],
                        'high':      float(candle_3['low']),
                        'low':       float(candle_1['high']),
                        'gap_size':  round(gap_size, 8),
                        'filled':    False,
                    })
                elif candle_1['low'] > candle_3['high']:
                    gap_size = candle_1['low'] - candle_3['high']
                    fvgs.append({
                        'type':      'FVG',
                        'direction': 'BEARISH',
                        'index':     i - 1,
                        'timestamp': data.index[i - 1],
                        'high':      float(candle_1['low']),
                        'low':       float(candle_3['high']),
                        'gap_size':  round(gap_size, 8),
                        'filled':    False,
                    })

            self.logger.info(
                "Fallback detected %d Fair Value Gaps.", len(fvgs))
            return fvgs

        except Exception as e:
            self.logger.error("Error detecting Fair Value Gaps: %s", e)
            return []
    
    # ==================== PHASE 2: STRUCTURE SHIFTS ====================
    
    def detect_market_structure_shift(
        self,
        data: pd.DataFrame,
        htf_trend: str,
        symbol: str = 'EURUSD',
    ) -> Optional[Dict]:
        """
        Detect Market Structure Shift (Change of Character / CHoCH) using
        the smartmoneyconcepts package bos_choch function.

        A Bearish MSS fires when HTF is BULLISH but a CHOCH to the downside
        appears, indicating the first internal low has been broken.
        A Bullish MSS fires when HTF is BEARISH but a CHOCH to the upside
        appears.

        Only fires against BULLISH or BEARISH trends. RANGING is excluded
        because there is no established structure to shift.

        Args:
            data:      H1 OHLCV DataFrame
            htf_trend: D1 trend string from determine_htf_trend()
            symbol:    Trading symbol for pip size calculation

        Returns:
            dict: MSS event dict if detected, None otherwise
        """
        if htf_trend not in ('BULLISH', 'BEARISH'):
            return None

        pip_size = utils.get_pip_value(symbol)
        if pip_size <= 0:
            pip_size = 0.0001

        current_price = float(data.iloc[-1]['close'])

        if _SMC_PKG_AVAILABLE and len(data) >= 30:
            try:
                swing_df = _smc_pkg.swing_highs_lows(data, swing_highs_lows=3)
                bc_df    = _smc_pkg.bos_choch(data, swing_df, close_break=True)

                # Walk CHOCH events from newest to oldest to find the most recent one.
                choch_rows = [
                    (idx, row)
                    for idx, row in bc_df.iterrows()
                    if not pd.isna(row.get('CHOCH'))
                ]

                for idx, row in reversed(choch_rows):
                    choch_val = int(row['CHOCH'])
                    level     = row.get('Level')
                    if pd.isna(level):
                        continue

                    level_price = float(level)

                    # Bearish CHOCH (-1) against a BULLISH HTF trend = Bearish MSS
                    if htf_trend == 'BULLISH' and choch_val == -1:
                        displacement_pips = abs(current_price - level_price) / pip_size
                        if displacement_pips < 5.0:
                            continue
                        self.logger.info(
                            "SMC package: Bearish MSS (CHoCH) at %.5f, "
                            "%.1f pip displacement.",
                            level_price, displacement_pips)
                        return {
                            'type':              'MSS',
                            'direction':         'BEARISH',
                            'level':             level_price,
                            'displacement':      displacement_pips,
                            'displacement_pips': displacement_pips,
                            'timestamp':         idx,
                        }

                    # Bullish CHOCH (1) against a BEARISH HTF trend = Bullish MSS
                    if htf_trend == 'BEARISH' and choch_val == 1:
                        displacement_pips = abs(current_price - level_price) / pip_size
                        if displacement_pips < 5.0:
                            continue
                        self.logger.info(
                            "SMC package: Bullish MSS (CHoCH) at %.5f, "
                            "%.1f pip displacement.",
                            level_price, displacement_pips)
                        return {
                            'type':              'MSS',
                            'direction':         'BULLISH',
                            'level':             level_price,
                            'displacement':      displacement_pips,
                            'displacement_pips': displacement_pips,
                            'timestamp':         idx,
                        }

                return None
            except Exception as e:
                self.logger.debug(
                    "SMC package MSS detection failed, using fallback: %s", e)

        # Fallback: internal swing break algorithm.
        try:
            recent_swings = self._identify_swings(data.tail(50), lookback=3)
            if len(recent_swings) < 3:
                return None

            if htf_trend == 'BULLISH':
                low_swings = [s for s in recent_swings if s['direction'] == 'LOW']
                if len(low_swings) < 2:
                    return None
                internal_low = float(low_swings[-1]['price'])
                if current_price < internal_low * 0.9995:
                    displacement_pips = abs(current_price - internal_low) / pip_size
                    if displacement_pips < 5.0:
                        return None
                    self.logger.info(
                        "Fallback Bearish MSS: price %.5f broke below %.5f "
                        "(%.1f pips).",
                        current_price, internal_low, displacement_pips)
                    return {
                        'type':              'MSS',
                        'direction':         'BEARISH',
                        'level':             internal_low,
                        'displacement':      displacement_pips,
                        'displacement_pips': displacement_pips,
                        'timestamp':         data.index[-1],
                    }
            else:
                high_swings = [s for s in recent_swings if s['direction'] == 'HIGH']
                if len(high_swings) < 2:
                    return None
                internal_high = float(high_swings[-1]['price'])
                if current_price > internal_high * 1.0005:
                    displacement_pips = abs(current_price - internal_high) / pip_size
                    if displacement_pips < 5.0:
                        return None
                    self.logger.info(
                        "Fallback Bullish MSS: price %.5f broke above %.5f "
                        "(%.1f pips).",
                        current_price, internal_high, displacement_pips)
                    return {
                        'type':              'MSS',
                        'direction':         'BULLISH',
                        'level':             internal_high,
                        'displacement':      displacement_pips,
                        'displacement_pips': displacement_pips,
                        'timestamp':         data.index[-1],
                    }

            return None

        except Exception as e:
            self.logger.error("Error detecting MSS: %s", e)
            return None
    
    def detect_break_of_structure(
        self,
        data: pd.DataFrame,
        htf_trend: str
    ) -> List[Dict]:
        """
        Detect Break of Structure (trend continuation) using the
        smartmoneyconcepts package bos_choch function.
        Falls back to the swing-based custom algorithm when the package
        is unavailable.

        Requires DOUBLE BOS for confirmation (len >= 2).

        Args:
            data:      OHLCV DataFrame
            htf_trend: 'BULLISH' or 'BEARISH'

        Returns:
            list: BOS event dicts with keys type, direction, level, timestamp
        """
        bos_events = []

        if _SMC_PKG_AVAILABLE and len(data) >= 30:
            try:
                swing_df = _smc_pkg.swing_highs_lows(data, swing_highs_lows=3)
                bc_df    = _smc_pkg.bos_choch(data, swing_df, close_break=True)

                for idx, row in bc_df.iterrows():
                    bos_val = row.get('BOS')
                    if pd.isna(bos_val):
                        continue

                    bos_int   = int(bos_val)
                    direction = 'BULLISH' if bos_int == 1 else 'BEARISH'

                    # Only return BOS events that match the HTF trend direction
                    if direction != htf_trend:
                        continue

                    level = row.get('Level')
                    if pd.isna(level):
                        continue

                    bos_events.append({
                        'type':      'BOS',
                        'direction': direction,
                        'level':     float(level),
                        'timestamp': idx,
                    })

                if len(bos_events) >= 2:
                    self.logger.info(
                        "SMC package: Double BOS confirmed for %s trend (%d events).",
                        htf_trend, len(bos_events))

                return bos_events
            except Exception as e:
                self.logger.debug(
                    "SMC package BOS detection failed, using fallback: %s", e)

        # Fallback: swing-based custom algorithm.
        try:
            tail_data      = data.tail(50)
            swings         = self._identify_swings(tail_data)
            tail_start_pos = len(data) - len(tail_data)

            if htf_trend == 'BULLISH':
                for i in range(len(swings) - 1):
                    if swings[i]['direction'] == 'HIGH':
                        swing_high    = swings[i]['price']
                        full_pos      = tail_start_pos + swings[i]['index']
                        later_candles = data.iloc[full_pos:]
                        if later_candles['close'].max() > swing_high:
                            bos_events.append({
                                'type':      'BOS',
                                'direction': 'BULLISH',
                                'level':     swing_high,
                                'timestamp': later_candles['close'].idxmax(),
                            })
            else:
                for i in range(len(swings) - 1):
                    if swings[i]['direction'] == 'LOW':
                        swing_low     = swings[i]['price']
                        full_pos      = tail_start_pos + swings[i]['index']
                        later_candles = data.iloc[full_pos:]
                        if later_candles['close'].min() < swing_low:
                            bos_events.append({
                                'type':      'BOS',
                                'direction': 'BEARISH',
                                'level':     swing_low,
                                'timestamp': later_candles['close'].idxmin(),
                            })

            if len(bos_events) >= 2:
                self.logger.info(
                    "Fallback: Double BOS confirmed for %s trend.", htf_trend)

            return bos_events

        except Exception as e:
            self.logger.error("Error detecting BOS: %s", e)
            return []
    
    # ==================== PHASE 3: INDUCEMENT & ENTRY ====================
    
    def detect_inducement(
        self,
        data: pd.DataFrame,
        direction: str,
        recent_swing: Dict
    ) -> Optional[Dict]:
        """
        Detect inducement (liquidity sweep) with quality validation (REFINEMENT #2).
        
        Args:
            data: OHLCV DataFrame
            direction: Expected direction after inducement
            recent_swing: Recent swing high/low that will be swept
            
        Returns:
            dict: Inducement information if detected
        """
        try:
            liquidity_level = recent_swing['price']
            recent_candles = data.tail(10)
            
            for i in range(len(recent_candles)):
                candle = recent_candles.iloc[i]
                
                if direction == 'BULLISH':
                    # Look for sweep below liquidity (stop hunt)
                    if candle['low'] <= liquidity_level:
                        # REFINEMENT #2: Validate inducement quality
                        wick_length = candle['close'] - candle['low']
                        body_size = abs(candle['close'] - candle['open'])
                        
                        # Wick swept, but body stayed above
                        if candle['close'] > liquidity_level:
                            sweep_pips = abs(candle['low'] - liquidity_level) / 0.0001
                            
                            # Check quality criteria
                            if config.INDUCEMENT_WICK_MIN_PIPS <= sweep_pips <= config.INDUCEMENT_WICK_MAX_PIPS:
                                if body_size > 0:
                                    body_ratio = body_size / (candle['high'] - candle['low'])
                                    
                                    if body_ratio >= config.INDUCEMENT_BODY_CLOSE_RATIO:
                                        return {
                                            'type': 'INDUCEMENT',
                                            'direction': direction,
                                            'liquidity_level': liquidity_level,
                                            'sweep_low': candle['low'],
                                            'candle_close': candle['close'],
                                            'sweep_pips': sweep_pips,
                                            'quality': 'STRONG',
                                            'timestamp': recent_candles.index[i]
                                        }
                
                else:  # BEARISH
                    # Look for sweep above liquidity
                    if candle['high'] >= liquidity_level:
                        wick_length = candle['high'] - candle['close']
                        body_size = abs(candle['close'] - candle['open'])
                        
                        if candle['close'] < liquidity_level:
                            sweep_pips = abs(candle['high'] - liquidity_level) / 0.0001
                            
                            if config.INDUCEMENT_WICK_MIN_PIPS <= sweep_pips <= config.INDUCEMENT_WICK_MAX_PIPS:
                                if body_size > 0:
                                    body_ratio = body_size / (candle['high'] - candle['low'])
                                    
                                    if body_ratio >= config.INDUCEMENT_BODY_CLOSE_RATIO:
                                        return {
                                            'type': 'INDUCEMENT',
                                            'direction': direction,
                                            'liquidity_level': liquidity_level,
                                            'sweep_high': candle['high'],
                                            'candle_close': candle['close'],
                                            'sweep_pips': sweep_pips,
                                            'quality': 'STRONG',
                                            'timestamp': recent_candles.index[i]
                                        }
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error detecting inducement: {e}")
            return None
    
    def calculate_entry_price(
        self,
        poi: Dict,
        setup_type: str,
        ml_score: int
    ) -> Dict:
        """
        Calculate precise entry using dynamic zones (REFINEMENT #1).
        
        Args:
            poi: Point of Interest (OB or BB)
            setup_type: 'UNICORN', 'OB', or 'BB'
            ml_score: ML confidence score
            
        Returns:
            dict: Entry price and configuration
        """
        try:
            poi_low   = float(poi['low'])
            poi_high  = float(poi['high'])
            poi_range = poi_high - poi_low
            direction = str(poi.get('direction', 'BULLISH')).upper()
            setup_u   = str(setup_type).upper()

            if poi_range <= 0:
                raise ValueError("Invalid POI range: high must be greater than low.")

            # Sniper entry model:
            # Lower zone for BUY (and upper zone for SELL) gives tighter risk and higher
            # potential RR, while still allowing confirmation mode for lower-quality setups.
            if setup_u == 'UNICORN' and ml_score >= 85:
                zone = 0.18
                entry_type = 'SNIPER'
                wait_confirmation = False
                expected_win_rate = 66
            elif setup_u == 'UNICORN' and ml_score >= config.ML_AUTO_EXECUTE_THRESHOLD:
                zone = 0.28
                entry_type = 'PRECISION'
                wait_confirmation = False
                expected_win_rate = 63
            elif ml_score >= 60:
                zone = 0.35
                entry_type = 'BALANCED'
                wait_confirmation = True
                expected_win_rate = 69
            else:
                zone = 0.42
                entry_type = 'CONFIRMATION'
                wait_confirmation = True
                expected_win_rate = 73

            # Limit order entry must be at the BOUNDARY where price FIRST ENTERS
            # the zone, not deep inside it.
            #
            # BUY setup:  price pulls BACK DOWN into a demand zone.
            #             It enters through the TOP (poi_high).
            #             Entry = poi_high - small zone offset.
            #             Result: order fills as soon as price touches the zone.
            #
            # SELL setup: price rallies UP into a supply zone.
            #             It enters through the BOTTOM (poi_low).
            #             Entry = poi_low + small zone offset.
            #             Result: order fills as soon as price touches the zone.
            #
            # Old code placed entry at the OPPOSITE end (deep inside zone).
            # That required price to travel through the entire zone before filling,
            # which rarely happens even within the 12-hour expiry window.
            if direction in ('BULLISH', 'BUY'):
                entry = poi_high - (poi_range * zone)
            else:
                entry = poi_low + (poi_range * zone)

            return {
                'entry_price': round(entry, 5),
                'entry_type': entry_type,
                'zone_percentage': round(zone * 100, 1),
                'wait_for_confirmation': wait_confirmation,
                'expected_win_rate': expected_win_rate,
                'expected_rr': round(1.0 / max(zone, 1e-6), 2),
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating entry price: {e}")
            # Fallback to 50% of POI
            return {
                'entry_price': round((poi['low'] + poi['high']) / 2, 5),
                'entry_type': 'BALANCED',
                'zone_percentage': 50.0,
                'wait_for_confirmation': True,
                'expected_win_rate': 70,
                'expected_rr': 2.0
            }
    
    def check_confirmation_candle(
        self,
        candle: Dict,
        poi: Dict,
        direction: str,
        avg_volume: float
    ) -> bool:
        """
        Validate confirmation candle before entry (REFINEMENT #2).
        
        Args:
            candle: Current candle data
            poi: Point of Interest
            direction: Expected direction
            avg_volume: Average volume for comparison
            
        Returns:
            bool: True if confirmation valid
        """
        try:
            poi_midpoint = (poi['low'] + poi['high']) / 2
            body_size = abs(candle['close'] - candle['open'])
            total_size = candle['high'] - candle['low']
            
            if total_size == 0:
                return False
            
            body_ratio = body_size / total_size
            
            if direction == 'BULLISH':
                is_bullish = candle['close'] > candle['open']
                close_above_mid = candle['close'] > poi_midpoint
                strong_body = body_ratio > config.CONFIRMATION_BODY_RATIO
                high_volume = candle['volume'] > avg_volume
                
                return all([is_bullish, close_above_mid, strong_body, high_volume])
            
            else:  # BEARISH
                is_bearish = candle['close'] < candle['open']
                close_below_mid = candle['close'] < poi_midpoint
                strong_body = body_ratio > config.CONFIRMATION_BODY_RATIO
                high_volume = candle['volume'] > avg_volume
                
                return all([is_bearish, close_below_mid, strong_body, high_volume])
        
        except Exception as e:
            self.logger.error(f"Error checking confirmation candle: {e}")
            return False
    
    # ==================== PHASE 4: RISK MANAGEMENT ====================
    
    def calculate_stop_loss(
        self,
        poi: Dict,
        direction: str,
        symbol: str,
        atr: float
    ) -> Dict:
        """
        Calculate stop loss with ATR adjustment (REFINEMENT #3).
        
        Args:
            poi: Point of Interest
            direction: Trade direction
            symbol: Trading symbol
            atr: Average True Range
            
        Returns:
            dict: Stop loss configuration
        """
        try:
            pip_size = utils.get_pip_value(symbol)
            if pip_size <= 0:
                raise ValueError("Invalid pip size for symbol.")

            direction_u = str(direction).upper()
            atr_pips    = (atr / pip_size) if atr and atr > 0 else 0.0

            # Dynamic structural buffer:
            # 20% of ATR, clamped to avoid both overly tight and overly wide stops.
            if atr_pips > 0:
                buffer_pips = min(8.0, max(2.0, atr_pips * 0.20))
            else:
                buffer_pips = 3.0

            if direction_u in ('BUY', 'BULLISH'):
                sl_price = float(poi['low']) - (buffer_pips * pip_size)
            else:  # SELL / BEARISH
                sl_price = float(poi['high']) + (buffer_pips * pip_size)
            
            return {
                'stop_loss': round(sl_price, 5),
                'buffer_pips': round(buffer_pips, 1),
                'atr_adjusted': True
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            # Fallback
            if str(direction).upper() in ('BUY', 'BULLISH'):
                return {'stop_loss': round(poi['low'] - (3 * utils.get_pip_value(symbol)), 5), 'buffer_pips': 3, 'atr_adjusted': False}
            else:
                return {'stop_loss': round(poi['high'] + (3 * utils.get_pip_value(symbol)), 5), 'buffer_pips': 3, 'atr_adjusted': False}
    
    def calculate_take_profits(
        self,
        entry: float,
        stop_loss: float,
        direction: str,
        htf_swing: float,
        symbol: str
    ) -> Dict:
        """
        Calculate exact TP1/TP2 risk multiples for live execution and training.

        Args:
            entry: Entry price
            stop_loss: Stop loss price
            direction: Trade direction
            htf_swing: HTF swing high/low
            symbol: Trading symbol
            
        Returns:
            dict: TP1 and TP2 levels
        """
        try:
            direction_u = str(direction).upper()
            risk_price  = abs(entry - stop_loss)
            risk_pips   = utils.calculate_pips(symbol, entry, stop_loss)
            if risk_pips <= 0:
                raise ValueError("Risk pips is zero - entry equals stop loss.")
            is_buy_dir = direction_u in ('BULLISH', 'BUY')
            target_tp1_rr = float(config.MIN_RR_RATIO)
            target_tp2_rr = float(config.MIN_RR_TP2)

            # Use the configured RR targets directly. Structure validates that
            # the setup has enough room for TP2, but it does not push the target
            # farther away than requested.
            if is_buy_dir:
                tp1 = entry + (risk_price * target_tp1_rr)
                tp2 = entry + (risk_price * target_tp2_rr)
            else:
                tp1 = entry - (risk_price * target_tp1_rr)
                tp2 = entry - (risk_price * target_tp2_rr)

            if htf_swing and htf_swing > 0:
                structural_pips = utils.calculate_pips(symbol, entry, htf_swing)
                structural_rr = (
                    round(structural_pips / risk_pips, 2) if risk_pips > 0 else 0.0
                )
                if is_buy_dir:
                    if htf_swing <= entry:
                        raise ValueError(
                            "HTF swing is not above entry for a bullish setup."
                        )
                    if structural_rr < target_tp2_rr:
                        raise ValueError(
                            "HTF swing R:R %.2f is below required TP2 %.1fR. "
                            "Setup rejected." % (structural_rr, target_tp2_rr)
                        )
                else:
                    if htf_swing >= entry:
                        raise ValueError(
                            "HTF swing is not below entry for a bearish setup."
                        )
                    if structural_rr < target_tp2_rr:
                        raise ValueError(
                            "HTF swing R:R %.2f is below required TP2 %.1fR. "
                            "Setup rejected." % (structural_rr, target_tp2_rr)
                        )

            tp1_pips = utils.calculate_pips(symbol, entry, tp1)
            tp2_pips = utils.calculate_pips(symbol, entry, tp2)
            tp1_rr   = round(tp1_pips / risk_pips, 2) if risk_pips > 0 else 0.0
            tp2_rr   = round(tp2_pips / risk_pips, 2) if risk_pips > 0 else 0.0

            return {
                'tp1':      round(tp1, 5),
                'tp2':      round(tp2, 5),
                'tp1_pips': round(tp1_pips, 1),
                'tp2_pips': round(tp2_pips, 1),
                'tp1_rr':   tp1_rr,
                'tp2_rr':   tp2_rr,
            }
        
        except ValueError:
            raise
        except Exception as e:
            self.logger.error("Error calculating take profits: %s", e)
            raise ValueError("TP calculation failed: %s" % e)
    
    # ==================== REFINEMENT FILTERS ====================
    
    def check_atr_filter(self, current_atr: float, atr_avg: float) -> Tuple[bool, str]:
        """
        REFINEMENT #5: ATR volatility filter.
        
        Args:
            current_atr: Current ATR value
            atr_avg: 20-period ATR average
            
        Returns:
            tuple: (pass_filter, reason)
        """
        volatility_ratio = current_atr / atr_avg if atr_avg > 0 else 1.0
        
        if volatility_ratio < config.ATR_MIN_RATIO:
            return False, f"Low volatility regime ({volatility_ratio:.2f}x) - skip trade"
        
        if volatility_ratio > config.ATR_MAX_RATIO:
            return False, f"High volatility regime ({volatility_ratio:.2f}x) - skip trade"
        
        return True, f"Normal volatility regime ({volatility_ratio:.2f}x)"
    
    def check_adx_filter(
        self,
        data: pd.DataFrame,
        period: int = 14
    ) -> Tuple[bool, float, str]:
        """
        ADX trend strength filter. ADX below 20 = ranging market = skip.

        Args:
            data:   OHLCV DataFrame (needs at least 2*period + 5 bars)
            period: ADX lookback period (default 14)

        Returns:
            tuple: (pass_filter, adx_value, reason)
        """
        try:
            n = len(data)
            if n < period * 2 + 5:
                return True, 0.0, "Insufficient data for ADX calculation"

            high  = data['high'].values.astype(float)
            low   = data['low'].values.astype(float)
            close = data['close'].values.astype(float)

            plus_dm  = np.zeros(n)
            minus_dm = np.zeros(n)
            tr_arr   = np.zeros(n)

            for i in range(1, n):
                h_diff        = high[i]    - high[i - 1]
                l_diff        = low[i - 1] - low[i]
                plus_dm[i]   = h_diff if (h_diff > l_diff and h_diff > 0) else 0.0
                minus_dm[i]  = l_diff if (l_diff > h_diff and l_diff > 0) else 0.0
                tr_arr[i]    = max(
                    high[i] - low[i],
                    abs(high[i]  - close[i - 1]),
                    abs(low[i]   - close[i - 1])
                )

            atr_s = np.zeros(n)
            pdm_s = np.zeros(n)
            mdm_s = np.zeros(n)

            atr_s[period] = tr_arr[1: period + 1].sum()
            pdm_s[period] = plus_dm[1: period + 1].sum()
            mdm_s[period] = minus_dm[1: period + 1].sum()

            for i in range(period + 1, n):
                atr_s[i] = atr_s[i - 1] - (atr_s[i - 1] / period) + tr_arr[i]
                pdm_s[i] = pdm_s[i - 1] - (pdm_s[i - 1] / period) + plus_dm[i]
                mdm_s[i] = mdm_s[i - 1] - (mdm_s[i - 1] / period) + minus_dm[i]

            pdi = np.zeros(n, dtype=float)
            mdi = np.zeros(n, dtype=float)
            valid_atr = atr_s > 0
            np.divide(100.0 * pdm_s, atr_s, out=pdi, where=valid_atr)
            np.divide(100.0 * mdm_s, atr_s, out=mdi, where=valid_atr)

            dx = np.zeros(n, dtype=float)
            pdi_mdi_sum = pdi + mdi
            valid_dx = pdi_mdi_sum > 0
            np.divide(
                100.0 * np.abs(pdi - mdi),
                pdi_mdi_sum,
                out=dx,
                where=valid_dx
            )

            adx    = np.zeros(n)
            start  = period * 2
            if start >= n:
                return True, 0.0, "Insufficient data for ADX smoothing"

            adx[start] = dx[period + 1: start + 1].mean()
            for i in range(start + 1, n):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

            adx_val = float(adx[-1]) if adx[-1] > 0 else float(dx[-1])

            if adx_val >= 25:
                return True,  adx_val, f"ADX {adx_val:.1f} - trending market"
            elif adx_val >= 20:
                return True,  adx_val, f"ADX {adx_val:.1f} - weak trend, proceed with caution"
            else:
                return False, adx_val, f"ADX {adx_val:.1f} - ranging market, skip"

        except Exception as e:
            self.logger.error("Error calculating ADX: %s", e)
            return True, 0.0, f"ADX calculation error: {e}"
    
    def check_session_filter(self, timestamp: datetime) -> Tuple[bool, str]:
        """
        REFINEMENT #6: Trading session filter.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            tuple: (pass_filter, reason)
        """
        utc_hour = timestamp.hour if hasattr(timestamp, 'hour') else datetime.utcnow().hour
        session  = utils.get_session_name(utc_hour)

        if config.AVOID_ASIAN_SESSION and session == 'Asian':
            return False, "Asian session - lower liquidity"

            # London/New York overlap window: 13:00-16:00 UTC
        is_overlap = 13 <= utc_hour < 16
        if config.PREFER_LONDON_NY_OVERLAP and is_overlap:
            return True, "London/NY overlap - optimal liquidity"

        if session in ('London', 'New York', 'Overlap', 'London Open'):
            return True, "%s session - acceptable liquidity" % session

        return False, "%s session - insufficient liquidity" % session
    
    def check_correlation_filter(
        self,
        symbol: str,
        direction: str,
        open_positions: List[Dict]
    ) -> Tuple[bool, str]:
        """
        REFINEMENT #7: Correlation exposure filter.
        
        Args:
            symbol: Symbol to trade
            direction: Trade direction
            open_positions: Currently open positions
            
        Returns:
            tuple: (pass_filter, reason)
        """
        # Define correlated pairs
        correlations = {
            'EURUSD': ['GBPUSD', 'AUDUSD', 'NZDUSD'],
            'GBPUSD': ['EURUSD', 'AUDUSD', 'NZDUSD'],
            'USDJPY': ['EURJPY', 'GBPJPY', 'AUDJPY'],
            'XAUUSD': ['XAGUSD'],
        }
        
        correlated_pairs = correlations.get(symbol, [])
        
        # Count correlated positions in same direction
        same_direction_count = sum(
            1 for pos in open_positions
            if pos['symbol'] in correlated_pairs and pos['direction'] == direction
        )
        
        if same_direction_count >= config.MAX_CORRELATED_POSITIONS:
            return False, f"Max correlated exposure reached ({same_direction_count}/{config.MAX_CORRELATED_POSITIONS})"
        
        return True, "Correlation check passed"
    
    def check_drawdown_filter(
        self,
        current_balance: float,
        peak_balance: float,
        base_risk: float
    ) -> Tuple[bool, float, str]:
        """
        REFINEMENT #8: Adaptive risk during drawdown.
        
        Args:
            current_balance: Current account balance
            peak_balance: Historical peak balance
            base_risk: Base risk percentage
            
        Returns:
            tuple: (allow_trade, adjusted_risk, reason)
        """
        drawdown_pct = ((peak_balance - current_balance) / peak_balance) * 100
        
        if drawdown_pct < 3:
            return True, base_risk, "No significant drawdown"
        
        elif drawdown_pct < 5:
            adjusted_risk = base_risk * 0.7
            return True, adjusted_risk, f"Moderate drawdown ({drawdown_pct:.1f}%) - reduced risk by 30%"
        
        elif drawdown_pct < 8:
            adjusted_risk = base_risk * 0.5
            return True, adjusted_risk, f"Significant drawdown ({drawdown_pct:.1f}%) - reduced risk by 50%"
        
        else:
            return False, 0, f"Severe drawdown ({drawdown_pct:.1f}%) - TRADING HALTED"
    
    # ==================== HELPER FUNCTIONS ====================
    
    def _calculate_ob_confidence(
        self,
        volume_ratio: float,
        impulse_volume: float,
        impulse_pips: float
    ) -> int:
        """
        Calculate confidence score for Order Block.
        
        Args:
            volume_ratio: OB candle volume / average volume
            impulse_volume: Impulse volume / average volume
            impulse_pips: Impulse move size in pips
            
        Returns:
            int: Confidence score 0-100
        """
        score = 0
        
        # Volume component (40 points)
        if volume_ratio >= 2.0:
            score += 20
        elif volume_ratio >= 1.5:
            score += 15
        else:
            score += 10
        
        if impulse_volume >= 2.5:
            score += 20
        elif impulse_volume >= 2.0:
            score += 15
        else:
            score += 10
        
        # Impulse size component (40 points)
        if impulse_pips >= 50:
            score += 40
        elif impulse_pips >= 30:
            score += 30
        else:
            score += 20
        
        # Base score (20 points)
        score += 20
        
        return min(score, 100)
    
    def detect_premium_discount_zone(
        self,
        current_price: float,
        swing_high: float,
        swing_low: float,
        direction: str
    ) -> Tuple[bool, float, str]:
        """
        Validate price is in the correct zone before entry.
        Institutions BUY from the discount half and SELL from the premium half.

        Returns:
            tuple: (is_in_correct_zone, zone_position_0_to_1, description)
        """
        try:
            if swing_high <= swing_low or swing_high <= 0 or swing_low <= 0:
                return True, 0.5, "Cannot determine zone - insufficient swing data"

            total_range    = swing_high - swing_low
            price_position = (current_price - swing_low) / total_range

            if direction == 'BULLISH':
                if price_position <= 0.35:
                    return True,  price_position, f"Deep discount zone ({price_position:.1%})"
                elif price_position <= 0.50:
                    return True,  price_position, f"Discount zone ({price_position:.1%})"
                elif price_position <= 0.65:
                    return False, price_position, f"Mid zone ({price_position:.1%}) - marginal"
                else:
                    return False, price_position, f"Premium zone ({price_position:.1%}) - avoid BUY"
            else:
                if price_position >= 0.65:
                    return True,  price_position, f"Deep premium zone ({price_position:.1%})"
                elif price_position >= 0.50:
                    return True,  price_position, f"Premium zone ({price_position:.1%})"
                elif price_position >= 0.35:
                    return False, price_position, f"Mid zone ({price_position:.1%}) - marginal"
                else:
                    return False, price_position, f"Discount zone ({price_position:.1%}) - avoid SELL"

        except Exception as e:
            self.logger.error("Error detecting premium/discount zone: %s", e)
            return True, 0.5, f"Zone detection error: {e}"

    def score_setup_quality(
        self,
        data: pd.DataFrame,
        poi: Dict,
        htf_trend: Dict,
        bos_events: List[Dict],
        direction: str,
        symbol: str
    ) -> int:
        """
        Composite 0-100 quality score for a setup.
        Used by both the live scheduler and the ML training pipeline
        as a pre-filter (only setups >= 50 enter training).

        Breakdown:
          HTF confidence  : 0-20 pts
          BOS count       : 0-15 pts
          POI volume      : 0-20 pts
          POI type        : 0-15 pts
          Zone correct    : 0-10 pts
          ADX strength    : 0-10 pts
          POI freshness   : 0-10 pts
        """
        score = 0
        try:
            # HTF confidence
            htf_confidence = int(htf_trend.get('confidence', 0))
            if htf_trend.get('trend') == direction:
                if htf_confidence >= 80:   score += 20
                elif htf_confidence >= 70: score += 16
                elif htf_confidence >= 60: score += 12
                else:                      score += 7

            # BOS count
            if len(bos_events) >= 3:   score += 15
            elif len(bos_events) == 2: score += 12
            elif len(bos_events) == 1: score += 6

            # Volume at POI
            vol_ratio = float(poi.get('volume_ratio', 0))
            if vol_ratio >= 3.0:   score += 20
            elif vol_ratio >= 2.5: score += 16
            elif vol_ratio >= 2.0: score += 12
            elif vol_ratio >= 1.5: score += 8
            elif vol_ratio >= 1.0: score += 4

            # POI type
            poi_type = str(poi.get('type', 'OB')).upper()
            if poi_type == 'UNICORN':             score += 15
            elif poi_type in ('BB', 'BREAKER'):   score += 12
            elif poi_type == 'OB':                score += 8
            elif poi_type == 'FVG':               score += 5

            # Premium/discount zone
            swing_high    = float(htf_trend.get('swing_high', 0))
            swing_low     = float(htf_trend.get('swing_low',  0))
            current_price = float(data.iloc[-1]['close'])
            if swing_high > swing_low > 0:
                in_zone, _, _ = self.detect_premium_discount_zone(
                    current_price, swing_high, swing_low, direction)
                if in_zone:
                    score += 10

            # ADX
            try:
                adx_ok, adx_val, _ = self.check_adx_filter(data.tail(60))
                if adx_val >= 35:    score += 10
                elif adx_val >= 30:  score += 8
                elif adx_val >= 25:  score += 6
                elif adx_ok:         score += 3
            except Exception:
                score += 5

            # Freshness
            bars_ago = max(0, len(data) - int(poi.get('index', len(data) - 1)))
            if bars_ago <= 5:    score += 10
            elif bars_ago <= 15: score += 8
            elif bars_ago <= 30: score += 5
            elif bars_ago <= 50: score += 2

        except Exception as e:
            self.logger.error("Error scoring setup quality: %s", e)

        return min(int(score), 100)
    
    def detect_unicorn_setup(
        self,
        breakers: List[Dict],
        fvgs: List[Dict],
        tolerance_pips: float = 0.0,
        atr_price: float = 0.0,
        symbol: str = 'EURUSD',
    ) -> Optional[Dict]:
        """
        Detect Unicorn setup (BB + FVG confluence).

        The overlap tolerance is ATR-relative when atr_price is supplied.
        This prevents tight instruments from never matching and volatile
        instruments like XAUUSD from matching zones that are too far apart.

        Args:
            breakers:       List of Breaker Blocks
            fvgs:           List of Fair Value Gaps
            tolerance_pips: Override tolerance in pips (0 = use ATR formula)
            atr_price:      ATR in price units from _calculate_atr()
            symbol:         Trading symbol for correct pip size

        Returns:
            dict: Unicorn setup if found, None otherwise
        """
        pip_size = utils.get_pip_value(symbol)
        if pip_size <= 0:
            pip_size = 0.0001

        # ATR-relative tolerance
        if tolerance_pips <= 0.0:
            if atr_price > 0.0:
                atr_pips      = atr_price / pip_size
                tolerance_pips = max(
                    config.UNICORN_TOLERANCE_MIN_PIPS,
                    atr_pips * config.UNICORN_TOLERANCE_ATR_MULT,
                )
            else:
                tolerance_pips = 10.0   # safe fallback before ATR data is available

        for bb in breakers:
            for fvg in fvgs:
                if bb['direction'] != fvg['direction']:
                    continue

                bb_mid   = (float(bb['high']) + float(bb['low'])) / 2.0
                fvg_mid  = (float(fvg['high']) + float(fvg['low'])) / 2.0
                distance = abs(bb_mid - fvg_mid) / pip_size

                if distance <= tolerance_pips:
                    return {
                        'type':         'UNICORN',
                        'breaker':      bb,
                        'fvg':          fvg,
                        'confidence':   95,
                        'direction':    bb['direction'],
                        'high':         max(float(bb['high']), float(fvg['high'])),
                        'low':          min(float(bb['low']),  float(fvg['low'])),
                        'volume_ratio': float(bb.get('volume_ratio', 1.5)),
                        'impulse_pips': float(bb.get('impulse_pips', 20.0)),
                        'timestamp':    bb.get('timestamp'),
                        'index':        bb.get('index', 0),
                    }

        return None
    
    def detect_inducement_post_structure(
        self,
        data: pd.DataFrame,
        poi: Dict,
        direction: str,
        lookback_bars: int = 40,
        symbol: str = 'EURUSD',
        timeframe: str = 'M15',
    ) -> Optional[Dict]:
        """
        Verify a liquidity sweep (inducement) has occurred on M15 AFTER
        the BOS/MSS and BEFORE price enters the POI.

        BUY: a candle must have wicked below an internal swing low and
             closed back above it (stop hunt complete).
        SELL: a candle must have wicked above an internal swing high and
              closed back below it.

        Returns dict if confirmed, None if sweep not yet happened.
        Caller should skip broadcast and retry on next scan cycle.
        """
        try:
            df = data.tail(lookback_bars).copy()

            if len(df) < 10:
                # Insufficient bars to confirm a liquidity sweep.
                # Return None so the scheduler's inducement guard fires correctly
                # and the setup is deferred to the next scan cycle.
                # Returning a non-None dict here was bypassing the guard and
                # broadcasting setups with zero sweep confirmation.
                self.logger.info(
                    "Insufficient M15 bars (%d) for inducement check. "
                    "Deferring setup until more bars are available.", len(df))
                return None

            is_buy   = direction == 'BULLISH'
            poi_high = float(poi.get('high', 0))
            poi_low  = float(poi.get('low',  0))

            if poi_high <= poi_low:
                self.logger.info(
                    "Malformed POI [%.5f - %.5f]. Passing inducement check.",
                    poi_low, poi_high)
                return None

            internal_swings = self._identify_swings(df, lookback=2)

            if not internal_swings:
                self.logger.info(
                    "No internal swings found in %d M15 bars.", len(df))
                return None

            if is_buy:
                # BUY demand zone is BELOW current price.
                # Smart money sweeps SELL STOPS sitting below the demand zone
                # (below poi_low) before reversing up into the zone.
                # We look for swing LOWS that are BELOW poi_low.
                # Old code looked for swing lows ABOVE poi_high — completely wrong.
                candidates = [
                    s for s in internal_swings
                    if s['direction'] == 'LOW' and s['price'] < poi_low
                ]
            else:
                # SELL supply zone is ABOVE current price.
                # Smart money sweeps BUY STOPS sitting above the supply zone
                # (above poi_high) before reversing down into the zone.
                # We look for swing HIGHS that are ABOVE poi_high.
                # Old code looked for swing highs BELOW poi_low — completely wrong.
                candidates = [
                    s for s in internal_swings
                    if s['direction'] == 'HIGH' and s['price'] > poi_high
                ]

            if not candidates:
                self.logger.info(
                    "No inducement candidates outside POI [%.5f - %.5f] "
                    "for %s.", poi_low, poi_high, direction)
                return None

            # Prefer the NEAREST valid swing to the POI boundary.
            # Nearest = smallest absolute distance from the zone edge.
            # This selects the most recently formed liquidity level
            # which is the most likely target for the stop hunt.
            _pip_size_for_dist = utils.get_pip_value(symbol)
            if _pip_size_for_dist <= 0:
                _pip_size_for_dist = 0.0001

            # Timeframe-scaled minimum distance.
            # A 3-pip sweep on M1 is proportionally the same as a 48-pip
            # sweep on D1. Scale by the timeframe multiplier so the threshold
            # is always meaningful relative to typical bar ranges.
            _tf_scale      = config.INDUCEMENT_TIMEFRAME_SCALE.get(
                timeframe.upper(), 1.5)
            _min_dist_pips = config.INDUCEMENT_MIN_PIPS_BASE * _tf_scale

            if is_buy:
                # For BUY: valid swing lows must be at least 3 pips BELOW poi_low
                valid_candidates = [
                    s for s in candidates
                    if (poi_low - s['price']) / _pip_size_for_dist >= _min_dist_pips
                ]
            else:
                # For SELL: valid swing highs must be at least 3 pips ABOVE poi_high
                valid_candidates = [
                    s for s in candidates
                    if (s['price'] - poi_high) / _pip_size_for_dist >= _min_dist_pips
                ]

            if not valid_candidates:
                self.logger.info(
                    "No qualifying inducement candidates found outside POI "
                    "[%.5f - %.5f] for %s with minimum %.1f pip distance. "
                    "Awaiting formation of a valid liquidity pool.",
                    poi_low, poi_high, direction, _min_dist_pips)
                return None

            # Select the nearest valid candidate to the zone boundary.
            if is_buy:
                target_swing = max(valid_candidates, key=lambda s: s['price'])
            else:
                target_swing = min(valid_candidates, key=lambda s: s['price'])

            sweep_level  = target_swing['price']
            swing_idx    = target_swing['index']

            sweep_candle = None
            _pip_size = utils.get_pip_value(symbol)
            if _pip_size <= 0:
                _pip_size = 0.0001

            for j in range(swing_idx + 1, len(df)):
                candle = df.iloc[j]
                if is_buy:
                    if (float(candle['low']) < sweep_level
                            and float(candle['close']) > sweep_level):
                        sweep_pips = (sweep_level - float(candle['low'])) / _pip_size
                        if sweep_pips >= _min_dist_pips:
                            sweep_candle = {
                                'index':       j,
                                'timestamp':   df.index[j],
                                'sweep_level': round(sweep_level, 5),
                                'sweep_low':   round(float(candle['low']), 5),
                                'close':       round(float(candle['close']), 5),
                                'sweep_pips':  round(sweep_pips, 1),
                            }
                            break
                else:
                    if (float(candle['high']) > sweep_level
                            and float(candle['close']) < sweep_level):
                        sweep_pips = (float(candle['high']) - sweep_level) / _pip_size
                        if sweep_pips >= _min_dist_pips:
                            sweep_candle = {
                                'index':       j,
                                'timestamp':   df.index[j],
                                'sweep_level': round(sweep_level, 5),
                                'sweep_high':  round(float(candle['high']), 5),
                                'close':       round(float(candle['close']), 5),
                                'sweep_pips':  round(sweep_pips, 1),
                            }
                            break                                

            if sweep_candle is None:
                self.logger.info(
                    "Sweep NOT YET confirmed. Internal %s at %.5f not yet "
                    "swept on M15. Awaiting inducement.",
                    'LOW' if is_buy else 'HIGH', sweep_level)
                return None

            current_price = float(df.iloc[-1]['close'])
            if is_buy and current_price < poi_low:
                self.logger.info(
                    "Inducement found but price %.5f already below POI low "
                    "%.5f. Setup invalidated.", current_price, poi_low)
                return None
            if not is_buy and current_price > poi_high:
                self.logger.info(
                    "Inducement found but price %.5f already above POI high "
                    "%.5f. Setup invalidated.", current_price, poi_high)
                return None

            quality = 'STRONG' if sweep_candle['sweep_pips'] >= 10.0 else 'MODERATE'

            self.logger.info(
                "Inducement CONFIRMED [%s]: %.1f pip sweep %s internal %s "
                "at %.5f. Price %.5f approaching POI [%.5f - %.5f]. "
                "Quality: %s.",
                direction, sweep_candle['sweep_pips'],
                'below' if is_buy else 'above',
                'LOW' if is_buy else 'HIGH', sweep_level,
                current_price, poi_low, poi_high, quality,
            )

            return {
                'type': 'INDUCEMENT', 'direction': direction,
                'sweep_level': sweep_level, 'sweep_candle': sweep_candle,
                'sweep_pips': sweep_candle['sweep_pips'],
                'quality': quality, 'timestamp': sweep_candle['timestamp'],
            }

        except Exception as e:
            self.logger.error(
                "Error in detect_inducement_post_structure: %s", e)
            return None
        
    def is_poi_mitigated(
        self,
        poi: Dict,
        data: pd.DataFrame,
        touch_mitigation: bool = False,
        symbol: str = 'EURUSD',
    ) -> bool:
        """
        Check whether a Point of Interest has been invalidated by price action.

        touch_mitigation=False (default):
            Any close beyond the zone boundary = mitigated.
            Use for standard Order Blocks.

        touch_mitigation=True:
            Zone is only mitigated if price closes MORE than
            MITIGATION_TOUCH_BUFFER_PIPS beyond the boundary.
            A wick or tight close that immediately reverses does NOT
            invalidate the zone.
            Use for refined Breaker Blocks where a touch-and-reject is
            itself a confirmation of institutional activity, not a kill.

        Args:
            poi:               Point of Interest dict.
            data:              OHLCV DataFrame used when detecting the POI.
            touch_mitigation:  When True, apply pip buffer before declaring mitigated.
            symbol:            Trading symbol for correct pip size.

        Returns:
            True  = POI is mitigated (do NOT trade it).
            False = POI is still fresh and valid.
        """
        try:
            poi_index = int(poi.get('index', 0))
            poi_high  = float(poi.get('high', 0))
            poi_low   = float(poi.get('low',  0))
            direction = str(poi.get('direction', 'BULLISH')).upper()

            post_poi = data.iloc[poi_index + 1:]
            if post_poi.empty:
                return False

            if touch_mitigation:
                pip_size  = utils.get_pip_value(symbol)
                if pip_size <= 0:
                    pip_size = 0.0001
                buffer    = config.MITIGATION_TOUCH_BUFFER_PIPS * pip_size
                if direction in ('BULLISH', 'BUY'):
                    # Mitigated only if a candle closes more than buffer below the low
                    return bool((post_poi['close'] < (poi_low - buffer)).any())
                else:
                    return bool((post_poi['close'] > (poi_high + buffer)).any())
            else:
                if direction in ('BULLISH', 'BUY'):
                    return bool((post_poi['close'] < poi_low).any())
                else:
                    return bool((post_poi['close'] > poi_high).any())

        except Exception as e:
            self.logger.error("Error in is_poi_mitigated: %s", e)
            return False

    def get_closest_unmitigated_poi(
        self,
        pois: List[Dict],
        sweep_level: float,
        direction: str,
        data: pd.DataFrame,
    ) -> Optional[Dict]:
        """
        From a list of POIs, return the CLOSEST unmitigated zone to the
        inducement sweep level.

        Why closest? Smart money sets entries at the nearest available
        institutional zone to where liquidity was swept. A distant zone
        is less likely to be the real order origin.

        For BUY setups: find the unmitigated bullish POI whose MID is
        nearest to (and at or above) the sweep_level.

        For SELL setups: find the unmitigated bearish POI whose MID is
        nearest to (and at or below) the sweep_level.

        Args:
            pois:        All POI candidates (already filtered for direction).
            sweep_level: Price level where the inducement sweep occurred.
            direction:   'BULLISH' or 'BEARISH'.
            data:        OHLCV DataFrame for mitigation checks.

        Returns:
            The closest valid POI dict, or None if nothing qualifies.
        """
        try:
            is_buy = direction.upper() in ('BULLISH', 'BUY')
            scored = []

            for p in pois:
                if self.is_poi_mitigated(p, data):
                    continue   # Already dug — skip

                p_mid  = (float(p.get('high', 0)) + float(p.get('low', 0))) / 2
                p_high = float(p.get('high', 0))
                p_low  = float(p.get('low',  0))

                if is_buy:
                    # Demand zone must sit at or above the sweep so price
                    # can return UP into it after the stop hunt.
                    if p_mid < sweep_level:
                        continue
                else:
                    # Supply zone must sit at or below the sweep so price
                    # can return DOWN into it after the stop hunt.
                    if p_mid > sweep_level:
                        continue

                distance = abs(p_mid - sweep_level)
                scored.append((distance, p))

            if not scored:
                return None

            # Sort ascending by distance — smallest distance first
            scored.sort(key=lambda x: x[0])
            best_distance, best_poi = scored[0]

            self.logger.info(
                "Closest unmitigated POI: [%.5f - %.5f] distance=%.5f from sweep %.5f.",
                float(best_poi.get('low', 0)), float(best_poi.get('high', 0)),
                best_distance, sweep_level,
            )
            return best_poi

        except Exception as e:
            self.logger.error("Error in get_closest_unmitigated_poi: %s", e)
            return None

    def is_nested_refinement_poi(
        self,
        parent_poi: Dict,
        child_poi: Dict,
        symbol: str = 'EURUSD',
        tolerance_pips: float = 0.25,
    ) -> bool:
        """
        Return True when the full lower-timeframe zone is contained inside the
        parent zone, allowing only a tiny tolerance for broker rounding.
        """
        try:
            parent_high = float(parent_poi.get('high', 0))
            parent_low  = float(parent_poi.get('low', 0))
            child_high  = float(child_poi.get('high', 0))
            child_low   = float(child_poi.get('low', 0))

            if parent_high <= parent_low or child_high <= child_low:
                return False

            pip_size = utils.get_pip_value(symbol)
            if pip_size <= 0:
                pip_size = 0.0001
            tol = max(0.0, float(tolerance_pips)) * pip_size
            parent_range = parent_high - parent_low
            child_range = child_high - child_low

            return (
                child_range < max(parent_range - tol, 0.0)
                and
                child_low >= (parent_low - tol)
                and child_high <= (parent_high + tol)
            )
        except Exception:
            return False

    def find_nested_refinement(
        self,
        parent_poi: Dict,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        htf_swing_high: float = 0.0,
        htf_swing_low: float = 0.0,
    ) -> Optional[Dict]:
        """
        Find the best lower-timeframe nested POI inside a parent H1/M15 zone.

        Preference order:
          1. Same POI type as the parent (BB inside BB, OB inside OB)
          2. Boundary closest to the parent entry edge
          3. Smallest fully nested zone for tighter risk
          4. Highest confidence
        """
        try:
            if data is None or len(data) < 30:
                return None

            direction   = str(parent_poi.get('direction', 'BULLISH')).upper()
            parent_type = str(parent_poi.get('type', 'OB')).upper()
            is_buy      = direction in ('BULLISH', 'BUY')
            data_tail   = data.tail(80)

            def _annotate(poi: Dict) -> Dict:
                out = dict(poi)
                out['timeframe'] = timeframe
                out['role'] = 'REFINEMENT'
                out['parent_timeframe'] = str(parent_poi.get('timeframe', 'H1'))
                return out

            candidate_groups: List[List[Dict]] = []
            if parent_type in ('BB', 'BREAKER'):
                candidate_groups.append([
                    _annotate(p) for p in self.detect_breaker_blocks(
                        data_tail,
                        direction,
                        htf_swing_high,
                        htf_swing_low,
                        symbol=symbol,
                    )
                ])
                candidate_groups.append([
                    _annotate(p) for p in self.detect_order_blocks(
                        data_tail,
                        direction,
                        symbol=symbol,
                    )
                ])
            else:
                candidate_groups.append([
                    _annotate(p) for p in self.detect_order_blocks(
                        data_tail,
                        direction,
                        symbol=symbol,
                    )
                ])
                candidate_groups.append([
                    _annotate(p) for p in self.detect_breaker_blocks(
                        data_tail,
                        direction,
                        htf_swing_high,
                        htf_swing_low,
                        symbol=symbol,
                    )
                ])

            nested: List[Dict] = []
            seen = set()
            for group in candidate_groups:
                for candidate in group[:12]:
                    if not self.is_nested_refinement_poi(
                        parent_poi,
                        candidate,
                        symbol=symbol,
                    ):
                        continue
                    key = (
                        str(candidate.get('type', '')),
                        round(float(candidate.get('low', 0)), 8),
                        round(float(candidate.get('high', 0)), 8),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    nested.append(candidate)

            if not nested:
                return None

            parent_entry_edge = float(
                parent_poi.get('high', 0) if is_buy else parent_poi.get('low', 0)
            )
            parent_is_breaker = parent_type in ('BB', 'BREAKER')

            def _sort_key(candidate: Dict):
                candidate_type = str(candidate.get('type', 'OB')).upper()
                candidate_is_breaker = candidate_type in ('BB', 'BREAKER')
                same_type_penalty = 0 if candidate_is_breaker == parent_is_breaker else 1
                entry_edge = float(
                    candidate.get('high', 0) if is_buy else candidate.get('low', 0)
                )
                boundary_distance = abs(parent_entry_edge - entry_edge)
                zone_range = abs(float(candidate.get('high', 0)) - float(candidate.get('low', 0)))
                confidence_penalty = -float(candidate.get('confidence', 0))
                return (
                    same_type_penalty,
                    boundary_distance,
                    zone_range,
                    confidence_penalty,
                )

            nested.sort(key=_sort_key)
            best = nested[0]
            best_range_pips = utils.calculate_pips(
                symbol,
                float(best.get('high', 0)),
                float(best.get('low', 0)),
            )
            self.logger.info(
                "Nested %s refinement selected for %s: %s [%.5f - %.5f] inside %s [%.5f - %.5f] (%.1f pips).",
                timeframe,
                symbol,
                str(best.get('type', 'OB')).upper(),
                float(best.get('low', 0)),
                float(best.get('high', 0)),
                str(parent_poi.get('timeframe', 'H1')).upper(),
                float(parent_poi.get('low', 0)),
                float(parent_poi.get('high', 0)),
                best_range_pips,
            )
            return best

        except Exception as e:
            self.logger.error("Error in find_nested_refinement: %s", e)
            return None
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range.
        Used by the scheduler's filter check and stop loss calculation.

        Args:
            data: OHLCV DataFrame
            period: ATR lookback period (default 14)

        Returns:
            float: ATR value, or 0.0 if insufficient data
        """
        try:
            if len(data) < period + 1:
                return 0.0

            high  = data['high']
            low   = data['low']
            close = data['close'].shift(1)

            tr1 = high - low
            tr2 = (high - close).abs()
            tr3 = (low  - close).abs()

            tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]

            return float(atr) if not pd.isna(atr) else 0.0

        except Exception as e:
            self.logger.error("Error calculating ATR: %s", e)
            return 0.0
