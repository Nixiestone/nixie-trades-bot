import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
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
        """Initialize SMC Strategy Engine."""
        self.logger = logging.getLogger(f"{__name__}.SMCStrategy")
        self.logger.info("SMC Strategy Engine initialized")
    
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
            
            # Count Higher Highs vs Lower Lows
            hh_count = sum(1 for s in swings if s['type'] == 'HH')
            ll_count = sum(1 for s in swings if s['type'] == 'LL')
            hl_count = sum(1 for s in swings if s['type'] == 'HL')
            lh_count = sum(1 for s in swings if s['type'] == 'LH')
            
            total = len(swings)
            bullish_ratio = (hh_count + hl_count) / total
            bearish_ratio = (ll_count + lh_count) / total
            
            if bullish_ratio >= 0.6:
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
            elif bearish_ratio >= 0.6:
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
        Identify swing highs and lows with trend classification.
        lookback=2 for HTF trend detection (finds more swings on Daily data).
        lookback=3 for entry-level structure (stricter, reduces noise on M15/H1).
        Uses ATR-based minimum size to reject noise spikes without losing real swings.
        """
        swings = []
        highs  = data['high'].values
        lows   = data['low'].values
        n      = len(data)

        close     = data['close'].values
        tr_values = []
        for k in range(1, n):
            tr = max(highs[k] - lows[k],
                     abs(highs[k] - close[k - 1]),
                     abs(lows[k]  - close[k - 1]))
            tr_values.append(tr)
        atr_approx     = float(np.mean(tr_values[-20:])) if len(tr_values) >= 20 else 0.0
        min_swing_size = atr_approx * 0.3

        for i in range(lookback, n - lookback):
            left_ok  = all(highs[i] > highs[i - j] for j in range(1, lookback + 1))
            right_ok = all(highs[i] > highs[i + j] for j in range(1, lookback + 1))
            if left_ok and right_ok:
                swing_size = highs[i] - min(lows[max(0, i - lookback): i + lookback + 1])
                if swing_size >= min_swing_size:
                    swings.append({
                        'index': i, 'price': highs[i],
                        'direction': 'HIGH', 'type': None
                    })

        for i in range(lookback, n - lookback):
            left_ok  = all(lows[i] < lows[i - j] for j in range(1, lookback + 1))
            right_ok = all(lows[i] < lows[i + j] for j in range(1, lookback + 1))
            if left_ok and right_ok:
                swing_size = max(highs[max(0, i - lookback): i + lookback + 1]) - lows[i]
                if swing_size >= min_swing_size:
                    swings.append({
                        'index': i, 'price': lows[i],
                        'direction': 'LOW', 'type': None
                    })

        swings.sort(key=lambda x: x['index'])

        # Compare like-to-like: HIGH vs previous HIGH, LOW vs previous LOW.
        # The old code compared every swing against the immediately previous one
        # regardless of type. A HIGH vs a LOW comparison is always True for HH
        # and always True for LL, producing a permanent 50/50 split that resolved
        # to RANGING for every symbol every time.
        last_high_price: Optional[float] = None
        last_low_price:  Optional[float] = None

        for swing in swings:
            if swing['direction'] == 'HIGH':
                if last_high_price is not None:
                    swing['type'] = 'HH' if swing['price'] > last_high_price else 'LH'
                else:
                    swing['type'] = 'HH'
                last_high_price = swing['price']
            else:
                if last_low_price is not None:
                    swing['type'] = 'LL' if swing['price'] < last_low_price else 'HL'
                else:
                    swing['type'] = 'LL'
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
        Detect Order Blocks with volume confirmation (REFINEMENT #1).
        
        Args:
            data: OHLCV DataFrame
            direction: 'BULLISH' or 'BEARISH'
            min_impulse_pips: Minimum impulse size in pips
            
        Returns:
            list: Valid Order Blocks with confidence scores
        """
        order_blocks = []
        
        try:
            # Pre-compute a rolling 20-bar average volume aligned to each candle.
            # Using data.tail(20).mean() for all candles is look-ahead bias:
            # a candle from 400 bars ago would be compared against future volume.
            # min_periods=5 ensures we still get a value for early candles.
            rolling_avg_vol = data['volume'].rolling(20, min_periods=5).mean()

            for i in range(10, len(data) - 6):
                candle       = data.iloc[i]
                next_candles = data.iloc[i+1:i+6]

                # Use the rolling average up to and including this candle.
                # If still NaN (first few bars), fall back to the candle's own volume.
                avg_volume = float(rolling_avg_vol.iloc[i])
                if avg_volume != avg_volume or avg_volume <= 0:
                    # NaN check: NaN != NaN is True in Python.
                    # Also guard against zero-volume brokers.
                    # Use a small positive sentinel so ratios are 1.0 (neutral).
                    avg_volume = max(float(candle['volume']), 1e-9)
                
                # Check for opposite-colored candle before impulse
                is_bullish_candle = candle['close'] > candle['open']
                is_bearish_candle = candle['close'] < candle['open']
                
                if direction == 'BULLISH' and not is_bearish_candle:
                    continue
                if direction == 'BEARISH' and not is_bullish_candle:
                    continue
                
                # Check for strong impulse after
                if direction == 'BULLISH':
                    impulse_move = next_candles['high'].max() - candle['low']
                else:
                    impulse_move = candle['high'] - next_candles['low'].min()
                
                # Convert to pips using the correct pip size for this symbol.
                # Using 0.0001 for all pairs overstates Gold pips by 1000x and
                # JPY pips by 100x, causing every OB on those instruments to be
                # incorrectly scored as having a massive impulse.
                _ob_pip_size  = utils.get_pip_value(symbol) if symbol else 0.0001
                if _ob_pip_size <= 0:
                    _ob_pip_size = 0.0001
                impulse_pips = impulse_move / _ob_pip_size
                
                if impulse_pips < min_impulse_pips:
                    continue
                
                # REFINEMENT #1: Volume confirmation
                # Guard against NaN and inf from zero-volume broker feeds.
                _candle_vol  = float(candle['volume'])
                _impulse_vol = float(next_candles['volume'].mean()) if len(next_candles) > 0 else 0.0
                volume_ratio   = _candle_vol  / avg_volume
                impulse_volume = _impulse_vol / avg_volume

                # NaN check: any comparison with NaN returns False, so we
                # must explicitly reject invalid ratios before the threshold check.
                if not (volume_ratio == volume_ratio) or volume_ratio <= 0:
                    continue
                if not (impulse_volume == impulse_volume) or impulse_volume <= 0:
                    continue

                if volume_ratio < config.VOLUME_MULTIPLIER_OB:
                    continue  # OB candle must have 1.5x volume

                if impulse_volume < config.VOLUME_MULTIPLIER_IMPULSE:
                    continue  # Impulse must have 2x volume
                
                # Reject OB candles with tiny bodies - they are indecision, not institutional
                candle_body_ratio = (abs(candle['close'] - candle['open'])
                                     / max(candle['high'] - candle['low'], 1e-9))
                if candle_body_ratio < 0.4:
                    continue
                
                # Valid Order Block found
                order_blocks.append({
                    'type': 'OB',
                    'direction': direction,
                    'index': i,
                    'timestamp': data.index[i],
                    'high': candle['high'],
                    'low': candle['low'],
                    'open': candle['open'],
                    'close': candle['close'],
                    'volume_ratio': volume_ratio,
                    'impulse_pips': impulse_pips,
                    'confidence': self._calculate_ob_confidence(
                        volume_ratio,
                        impulse_volume,
                        impulse_pips
                    )
                })
            
            # Sort by confidence
            order_blocks.sort(key=lambda x: x['confidence'], reverse=True)
            
            self.logger.info(f"Detected {len(order_blocks)} valid Order Blocks for {direction} direction")
            return order_blocks
        
        except Exception as e:
            self.logger.error(f"Error detecting Order Blocks: {e}")
            return []
    
    def detect_breaker_blocks(
        self,
        data: pd.DataFrame,
        direction: str,
        htf_swing_high: float,
        htf_swing_low: float
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
                            breakers.append({
                                'type': 'BB',
                                'direction': direction,
                                'index': i,
                                'timestamp': data.index[i],
                                'high': candle['high'],
                                'low': candle['low'],
                                'breaker_level': resistance,
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
                            breakers.append({
                                'type': 'BB',
                                'direction': direction,
                                'index': i,
                                'timestamp': data.index[i],
                                'high': candle['high'],
                                'low': candle['low'],
                                'breaker_level': support,
                                'confidence': 75
                            })
            
            self.logger.info(f"Detected {len(breakers)} valid Breaker Blocks for {direction} direction")
            return breakers
        
        except Exception as e:
            self.logger.error(f"Error detecting Breaker Blocks: {e}")
            return []
    
    def detect_fair_value_gaps(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect Fair Value Gaps (3-candle imbalances).
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            list: Fair Value Gaps
        """
        fvgs = []
        
        try:
            for i in range(2, len(data)):
                candle_1 = data.iloc[i-2]
                candle_2 = data.iloc[i-1]
                candle_3 = data.iloc[i]
                
                # Bullish FVG: candle_1.high < candle_3.low
                if candle_1['high'] < candle_3['low']:
                    gap_size = candle_3['low'] - candle_1['high']
                    
                    fvgs.append({
                        'type': 'FVG',
                        'direction': 'BULLISH',
                        'index': i-1,
                        'timestamp': data.index[i-1],
                        'high': candle_3['low'],
                        'low': candle_1['high'],
                        'gap_size': gap_size,
                        'filled': False
                    })
                
                # Bearish FVG: candle_1.low > candle_3.high
                elif candle_1['low'] > candle_3['high']:
                    gap_size = candle_1['low'] - candle_3['high']
                    
                    fvgs.append({
                        'type': 'FVG',
                        'direction': 'BEARISH',
                        'index': i-1,
                        'timestamp': data.index[i-1],
                        'high': candle_1['low'],
                        'low': candle_3['high'],
                        'gap_size': gap_size,
                        'filled': False
                    })
            
            self.logger.info(f"Detected {len(fvgs)} Fair Value Gaps")
            return fvgs
        
        except Exception as e:
            self.logger.error(f"Error detecting Fair Value Gaps: {e}")
            return []
    
    # ==================== PHASE 2: STRUCTURE SHIFTS ====================
    
    def detect_market_structure_shift(
        self,
        data: pd.DataFrame,
        htf_trend: str,
        symbol: str = 'EURUSD',
    ) -> Optional[Dict]:
        """
        Detect Market Structure Shift (potential reversal).

        A Bearish MSS fires when the HTF trend is BULLISH but price breaks
        below the MOST RECENT internal swing low — the first sign institutions
        are distributing and reversing direction.

        A Bullish MSS fires when the HTF trend is BEARISH but price breaks
        above the MOST RECENT internal swing high — the first sign of
        accumulation and a potential upside reversal.

        Only fires when HTF trend is clearly BULLISH or BEARISH. RANGING
        markets are excluded because there is no established structure to shift.

        Args:
            data:      H1 OHLCV DataFrame
            htf_trend: D1 trend string from determine_htf_trend()
            symbol:    Trading symbol for correct pip size calculation

        Returns:
            dict: MSS event dict if detected, None otherwise
        """
        try:
            # Only look for MSS against a clear directional trend.
            # A RANGING market has no established structure to shift.
            if htf_trend not in ('BULLISH', 'BEARISH'):
                return None

            recent_swings = self._identify_swings(data.tail(50), lookback=3)

            if len(recent_swings) < 3:
                return None

            current_price = float(data.iloc[-1]['close'])
            pip_size      = utils.get_pip_value(symbol)
            if pip_size <= 0:
                pip_size = 0.0001

            if htf_trend == 'BULLISH':
                # Bearish MSS: price breaks below the MOST RECENT internal swing low.
                # Using the most recent low (not the minimum) matches the SMC rule:
                # the first internal low that breaks signals a structure shift.
                low_swings = [
                    s for s in recent_swings if s['direction'] == 'LOW'
                ]
                if len(low_swings) < 2:
                    return None
                # Most recent internal low is the last one in time-sorted list
                internal_low = float(low_swings[-1]['price'])
                if current_price < internal_low * 0.9995:
                    displacement_pips = abs(current_price - internal_low) / pip_size
                    # Minimum 5-pip displacement to reject noise
                    if displacement_pips < 5.0:
                        return None
                    self.logger.info(
                        "Bearish MSS detected: price %.5f broke below "
                        "internal low %.5f (%.1f pips displacement).",
                        current_price, internal_low, displacement_pips,
                    )
                    return {
                        'type':              'MSS',
                        'direction':         'BEARISH',
                        'level':             internal_low,
                        'displacement':      displacement_pips,
                        'displacement_pips': displacement_pips,
                        'timestamp':         data.index[-1],
                    }

            else:  # BEARISH
                # Bullish MSS: price breaks above the MOST RECENT internal swing high.
                high_swings = [
                    s for s in recent_swings if s['direction'] == 'HIGH'
                ]
                if len(high_swings) < 2:
                    return None
                internal_high = float(high_swings[-1]['price'])
                if current_price > internal_high * 1.0005:
                    displacement_pips = abs(current_price - internal_high) / pip_size
                    if displacement_pips < 5.0:
                        return None
                    self.logger.info(
                        "Bullish MSS detected: price %.5f broke above "
                        "internal high %.5f (%.1f pips displacement).",
                        current_price, internal_high, displacement_pips,
                    )
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
        Detect Break of Structure (trend continuation).
        Requires DOUBLE BOS for confirmation.

        Args:
            data: OHLCV DataFrame
            htf_trend: Current HTF trend direction string

        Returns:
            list: BOS events
        """
        bos_events = []

        try:
            tail_data = data.tail(50)
            swings    = self._identify_swings(tail_data)

            # tail_data is a slice of data. swings[i]['index'] is positional
            # within tail_data. We need the positional offset in the full data
            # DataFrame so we can use iloc safely and avoid duplicate-label bugs
            # that occur when MT5 returns repeated timestamps at session boundaries.
            tail_start_pos = len(data) - len(tail_data)

            if htf_trend == 'BULLISH':
                for i in range(len(swings) - 1):
                    if swings[i]['direction'] == 'HIGH':
                        swing_high     = swings[i]['price']
                        full_pos       = tail_start_pos + swings[i]['index']
                        later_candles  = data.iloc[full_pos:]
                        if later_candles['close'].max() > swing_high:
                            bos_events.append({
                                'type':      'BOS',
                                'direction': 'BULLISH',
                                'level':     swing_high,
                                'timestamp': later_candles['close'].idxmax(),
                            })

            else:  # BEARISH
                for i in range(len(swings) - 1):
                    if swings[i]['direction'] == 'LOW':
                        swing_low      = swings[i]['price']
                        full_pos       = tail_start_pos + swings[i]['index']
                        later_candles  = data.iloc[full_pos:]
                        if later_candles['close'].min() < swing_low:
                            bos_events.append({
                                'type':      'BOS',
                                'direction': 'BEARISH',
                                'level':     swing_low,
                                'timestamp': later_candles['close'].idxmin(),
                            })

            if len(bos_events) >= 2:
                self.logger.info("Double BOS confirmed for %s trend", htf_trend)

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
            # which almost never happens within the 8-hour expiry window.
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
        Calculate TP1 and TP2 with Fibonacci extension (REFINEMENT #4).
        
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

            # If the structural swing target is not beyond entry in trade direction,
            # synthesize a minimum valid structural target from risk multiples.
            if direction_u in ('BULLISH', 'BUY'):
                if htf_swing <= entry:
                    htf_swing = entry + (risk_price * config.MIN_RR_TP2)
            else:
                if htf_swing >= entry:
                    htf_swing = entry - (risk_price * config.MIN_RR_TP2)

            # TP1: Internal structural target.
            # Use the greater of:
            # - Midpoint to HTF swing
            # - Minimum configured RR target
            if direction_u in ('BULLISH', 'BUY'):
                tp1_structural = entry + ((htf_swing - entry) * 0.5)
                tp1_min_rr     = entry + (risk_price * config.MIN_RR_RATIO)
                tp1            = max(tp1_structural, tp1_min_rr)
            else:
                tp1_structural = entry - ((entry - htf_swing) * 0.5)
                tp1_min_rr     = entry - (risk_price * config.MIN_RR_RATIO)
                tp1            = min(tp1_structural, tp1_min_rr)

            tp1_pips = utils.calculate_pips(symbol, entry, tp1)
            tp1_rr   = round(tp1_pips / risk_pips, 2) if risk_pips > 0 else 0.0

            if tp1_rr < config.MIN_RR_RATIO:
                raise ValueError(
                    "TP1 R:R %.2f is below minimum %.1f. "
                    "Structure does not offer sufficient reward. Setup rejected." % (
                        tp1_rr, config.MIN_RR_RATIO)
                )

            # TP2: Full external structural target - the complete HTF swing level.
            tp2      = htf_swing
            tp2_pips = utils.calculate_pips(symbol, entry, tp2)
            tp2_rr   = round(tp2_pips / risk_pips, 2) if risk_pips > 0 else 0.0

            # Sniper rule: do not force a reduced TP2.
            # If structure cannot provide required RR at TP2, reject the setup.
            if tp2_rr < config.MIN_RR_TP2:
                raise ValueError(
                    "TP2 R:R %.2f is below minimum %.1f. "
                    "No valid external target for sniper continuation." % (
                        tp2_rr, config.MIN_RR_TP2)
                )

            # Cap TPs at maximum RR so price has a realistic chance of reaching them
            # within the setup expiry window. The HTF D1 swing target is often
            # 400-700 pips away, producing RR of 1:50+ on tight H1 SLs. These
            # never fill and count as losses when the stop is hit on the reversal.
            _max_tp1_rr = getattr(config, 'MAX_RR_TP1', 5.0)
            _max_tp2_rr = getattr(config, 'MAX_RR_TP2', 10.0)
            _is_buy_dir = direction_u in ('BULLISH', 'BUY')

            _cap_tp1 = (entry + risk_price * _max_tp1_rr
                        if _is_buy_dir else entry - risk_price * _max_tp1_rr)
            _cap_tp2 = (entry + risk_price * _max_tp2_rr
                        if _is_buy_dir else entry - risk_price * _max_tp2_rr)

            if _is_buy_dir:
                tp1 = min(tp1, _cap_tp1)
                tp2 = min(tp2, _cap_tp2)
            else:
                tp1 = max(tp1, _cap_tp1)
                tp2 = max(tp2, _cap_tp2)

            # Ensure TP2 remains beyond TP1 after capping
            if _is_buy_dir and tp2 <= tp1:
                tp2 = tp1 + risk_price
            elif not _is_buy_dir and tp2 >= tp1:
                tp2 = tp1 - risk_price

            # Recalculate pips and RR after capping
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
        tolerance_pips: float = 10
    ) -> Optional[Dict]:
        """
        Detect Unicorn setup (BB + FVG confluence).
        
        Args:
            breakers: List of Breaker Blocks
            fvgs: List of Fair Value Gaps
            tolerance_pips: Maximum distance for overlap
            
        Returns:
            dict: Unicorn setup if found
        """
        for bb in breakers:
            for fvg in fvgs:
                # Check if same direction
                if bb['direction'] != fvg['direction']:
                    continue
                
                # Check for overlap
                bb_mid = (bb['high'] + bb['low']) / 2
                fvg_mid = (fvg['high'] + fvg['low']) / 2
                
                distance = abs(bb_mid - fvg_mid) / 0.0001  # Convert to pips
                
                if distance <= tolerance_pips:
                    return {
                        'type': 'UNICORN',
                        'breaker': bb,
                        'fvg': fvg,
                        'confidence': 95,
                        'direction': bb['direction'],
                        'high': max(bb['high'], fvg['high']),
                        'low': min(bb['low'], fvg['low'])
                    }
        
        return None
    
    def detect_inducement_post_structure(
        self,
        data: pd.DataFrame,
        poi: Dict,
        direction: str,
        lookback_bars: int = 40,
        symbol: str = 'EURUSD',
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

            # Minimum distance: the swing must be at least 3 pips outside the zone.
            # Anything closer is noise, not a real liquidity pool.
            _min_dist_pips = 3.0

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
                        if sweep_pips >= 3.0:
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
                        if sweep_pips >= 3.0:
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
        
    def is_poi_mitigated(self, poi: Dict, data: pd.DataFrame) -> bool:
        """
        Check whether a Point of Interest has been invalidated by price action.

        A bullish POI (demand zone) is mitigated when any candle CLOSES
        below the POI low after the zone was formed. Price returning to the
        zone and closing inside it means the buyers there were defeated.

        A bearish POI (supply zone) is mitigated when any candle CLOSES
        above the POI high after the zone was formed.

        Args:
            poi:  The POI dict (must have 'index', 'high', 'low', 'direction').
            data: The same OHLCV DataFrame used when detecting the POI.

        Returns:
            True  — POI is mitigated (do NOT trade it).
            False — POI is still fresh and valid.
        """
        try:
            poi_index = int(poi.get('index', 0))
            poi_high  = float(poi.get('high', 0))
            poi_low   = float(poi.get('low',  0))
            direction = str(poi.get('direction', 'BULLISH')).upper()

            # Only look at candles that formed AFTER the POI
            post_poi = data.iloc[poi_index + 1:]

            if post_poi.empty:
                return False   # No candles after POI yet — still fresh

            if direction in ('BULLISH', 'BUY'):
                # Mitigated if any close is below the demand zone low
                return bool((post_poi['close'] < poi_low).any())
            else:
                # Mitigated if any close is above the supply zone high
                return bool((post_poi['close'] > poi_high).any())

        except Exception as e:
            self.logger.error("Error in is_poi_mitigated: %s", e)
            return False   # On error, assume valid (conservative)

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
