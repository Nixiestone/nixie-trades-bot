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
                return {
                    'trend': 'BULLISH',
                    'confidence': int(bullish_ratio * 100),
                    'reason': f'{hh_count} Higher Highs confirmed',
                    'swing_high': swings[-1]['price'] if swings[-1]['type'] in ['HH', 'HL'] else swings[-2]['price'],
                    'swing_low': min(s['price'] for s in swings if s['type'] in ['HL', 'LL'])
                }
            elif bearish_ratio >= 0.6:
                return {
                    'trend': 'BEARISH',
                    'confidence': int(bearish_ratio * 100),
                    'reason': f'{ll_count} Lower Lows confirmed',
                    'swing_high': max(s['price'] for s in swings if s['type'] in ['LH', 'HH']),
                    'swing_low': swings[-1]['price'] if swings[-1]['type'] in ['LL', 'LH'] else swings[-2]['price']
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
        min_impulse_pips: float = 20
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
            # Calculate average volume (last 20 candles)
            avg_volume = data['volume'].tail(20).mean()
            
            for i in range(10, len(data) - 6):
                candle = data.iloc[i]
                next_candles = data.iloc[i+1:i+6]
                
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
                
                # Convert to pips (assuming EURUSD-like, adjust for symbol)
                impulse_pips = impulse_move / 0.0001
                
                if impulse_pips < min_impulse_pips:
                    continue
                
                # REFINEMENT #1: Volume confirmation
                volume_ratio = candle['volume'] / avg_volume
                impulse_volume = next_candles['volume'].mean() / avg_volume
                
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
            for i in range(20, len(data) - 5):
                candle = data.iloc[i]
                prev_candles = data.iloc[i-10:i]
                next_candles = data.iloc[i+1:i+5]
                
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
        htf_trend: str
    ) -> Optional[Dict]:
        """
        Detect Market Structure Shift (potential reversal).
        
        Args:
            data: OHLCV DataFrame
            htf_trend: Current HTF trend
            
        Returns:
            dict: MSS information if detected, None otherwise
        """
        try:
            # Look for displacement breaking internal structure
            recent_swings = self._identify_swings(data.tail(50), lookback=3)
            
            if len(recent_swings) < 3:
                return None
            
            current_price = data.iloc[-1]['close']
            
            if htf_trend == 'BULLISH':
                # Look for break of internal low (bearish MSS)
                internal_low = min(s['price'] for s in recent_swings[-5:] if s['direction'] == 'LOW')
                
                if current_price < internal_low * 0.998:  # Broke below with confirmation
                    return {
                        'type': 'MSS',
                        'direction': 'BEARISH',
                        'level': internal_low,
                        'displacement': abs(current_price - internal_low) / 0.0001,
                        'timestamp': data.index[-1]
                    }
            
            else:  # BEARISH
                # Look for break of internal high (bullish MSS)
                internal_high = max(s['price'] for s in recent_swings[-5:] if s['direction'] == 'HIGH')
                
                if current_price > internal_high * 1.002:
                    return {
                        'type': 'MSS',
                        'direction': 'BULLISH',
                        'level': internal_high,
                        'displacement': abs(current_price - internal_high) / 0.0001,
                        'timestamp': data.index[-1]
                    }
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error detecting MSS: {e}")
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

            if htf_trend == 'BULLISH':
                for i in range(len(swings) - 1):
                    if swings[i]['direction'] == 'HIGH':
                        swing_high  = swings[i]['price']
                        # Convert positional index within tail_data to an actual label
                        swing_label = tail_data.index[swings[i]['index']]
                        later_candles = data.loc[swing_label:]

                        if later_candles['close'].max() > swing_high:
                            bos_events.append({
                                'type':      'BOS',
                                'direction': 'BULLISH',
                                'level':     swing_high,
                                # idxmax() returns a label directly - do not wrap in .index[]
                                'timestamp': later_candles['close'].idxmax(),
                            })

            else:  # BEARISH
                for i in range(len(swings) - 1):
                    if swings[i]['direction'] == 'LOW':
                        swing_low   = swings[i]['price']
                        swing_label = tail_data.index[swings[i]['index']]
                        later_candles = data.loc[swing_label:]

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

            if direction in ('BULLISH', 'BUY'):
                entry = poi_low + (poi_range * zone)
            else:
                entry = poi_high - (poi_range * zone)

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
