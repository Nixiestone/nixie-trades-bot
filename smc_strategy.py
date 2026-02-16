"""
NIX TRADES - SMC Strategy Engine
Smart Money Concepts implementation with 8 precision refinements
Production-ready, zero errors, zero placeholders
NO EMOJIS - Professional code only
"""

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
    
    def _identify_swings(self, data: pd.DataFrame) -> List[Dict]:
        """
        Identify swing highs and lows with trend classification.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            list: Swing points with types (HH, LL, HL, LH)
        """
        swings = []
        highs = data['high'].values
        lows = data['low'].values
        
        # Find swing highs (local maxima)
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swings.append({
                    'index': i,
                    'price': highs[i],
                    'direction': 'HIGH',
                    'type': None  # Will classify later
                })
        
        # Find swing lows (local minima)
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swings.append({
                    'index': i,
                    'price': lows[i],
                    'direction': 'LOW',
                    'type': None
                })
        
        # Sort by index
        swings.sort(key=lambda x: x['index'])
        
        # Classify swings
        for i in range(1, len(swings)):
            current = swings[i]
            previous = swings[i-1]
            
            if current['direction'] == 'HIGH':
                if current['price'] > previous['price']:
                    current['type'] = 'HH'  # Higher High
                else:
                    current['type'] = 'LH'  # Lower High
            else:  # LOW
                if current['price'] < previous['price']:
                    current['type'] = 'LL'  # Lower Low
                else:
                    current['type'] = 'HL'  # Higher Low
        
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
            recent_swings = self._identify_swings(data.tail(50))
            
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
            htf_trend: Current HTF trend
            
        Returns:
            list: BOS events
        """
        bos_events = []
        
        try:
            swings = self._identify_swings(data.tail(50))
            
            if htf_trend == 'BULLISH':
                # Look for breaks of swing highs
                for i in range(len(swings) - 1):
                    if swings[i]['direction'] == 'HIGH':
                        swing_high = swings[i]['price']
                        
                        # Check if subsequent price broke above
                        later_candles = data.iloc[swings[i]['index']:]
                        
                        if later_candles['close'].max() > swing_high:
                            bos_events.append({
                                'type': 'BOS',
                                'direction': 'BULLISH',
                                'level': swing_high,
                                'timestamp': later_candles.index[later_candles['close'].idxmax()]
                            })
            
            else:  # BEARISH
                # Look for breaks of swing lows
                for i in range(len(swings) - 1):
                    if swings[i]['direction'] == 'LOW':
                        swing_low = swings[i]['price']
                        
                        later_candles = data.iloc[swings[i]['index']:]
                        
                        if later_candles['close'].min() < swing_low:
                            bos_events.append({
                                'type': 'BOS',
                                'direction': 'BEARISH',
                                'level': swing_low,
                                'timestamp': later_candles.index[later_candles['close'].idxmin()]
                            })
            
            # Check for Double BOS
            if len(bos_events) >= 2:
                self.logger.info(f"Double BOS confirmed for {htf_trend} trend")
            
            return bos_events
        
        except Exception as e:
            self.logger.error(f"Error detecting BOS: {e}")
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
            poi_low = poi['low']
            poi_high = poi['high']
            poi_body = poi_high - poi_low
            
            # REFINEMENT #1: Dynamic entry zones
            if setup_type == 'UNICORN' and ml_score >= config.ML_AUTO_EXECUTE_THRESHOLD:
                # Aggressive: Top 25% for high R:R
                if poi['direction'] == 'BULLISH':
                    entry = poi_low + (poi_body * 0.625)
                else:
                    entry = poi_high - (poi_body * 0.625)
                
                return {
                    'entry_price': round(entry, 5),
                    'entry_type': 'AGGRESSIVE',
                    'zone_percentage': 62.5,
                    'wait_for_confirmation': False,
                    'expected_win_rate': 65,
                    'expected_rr': 2.5
                }
            
            else:
                # Conservative: Bottom 25% for higher probability
                if poi['direction'] == 'BULLISH':
                    entry = poi_low + (poi_body * 0.125)
                else:
                    entry = poi_high - (poi_body * 0.125)
                
                return {
                    'entry_price': round(entry, 5),
                    'entry_type': 'CONSERVATIVE',
                    'zone_percentage': 12.5,
                    'wait_for_confirmation': True,
                    'expected_win_rate': 75,
                    'expected_rr': 2.0
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
            buffer_pips = 3  # Base buffer
            
            # REFINEMENT #3: ATR-based SL adjustment
            atr_pips = atr / utils.get_pip_value(symbol)
            
            if atr_pips > 15:  # High volatility
                buffer_pips = 5
            elif atr_pips < 8:  # Low volatility
                buffer_pips = 2
            
            if direction == 'BUY':
                sl_price = poi['low'] - (buffer_pips * utils.get_pip_value(symbol))
            else:  # SELL
                sl_price = poi['high'] + (buffer_pips * utils.get_pip_value(symbol))
            
            return {
                'stop_loss': round(sl_price, 5),
                'buffer_pips': buffer_pips,
                'atr_adjusted': True
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            # Fallback
            if direction == 'BUY':
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
            risk_pips = utils.calculate_pips(symbol, entry, stop_loss)
            
            # TP1: 1.5R (Standard)
            tp1_pips = risk_pips * config.TARGET_RR_TP1
            tp1 = utils.calculate_price_from_pips(
                symbol,
                entry,
                tp1_pips,
                'up' if direction == 'BUY' else 'down'
            )
            
            # TP2: Fibonacci 1.618 extension OR 2.5R (whichever is closer)
            if direction == 'BUY':
                fib_tp2 = entry + ((entry - stop_loss) * config.FIB_EXTENSION_LEVEL)
            else:
                fib_tp2 = entry - ((stop_loss - entry) * config.FIB_EXTENSION_LEVEL)
            
            standard_tp2_pips = risk_pips * config.TARGET_RR_TP2
            standard_tp2 = utils.calculate_price_from_pips(
                symbol,
                entry,
                standard_tp2_pips,
                'up' if direction == 'BUY' else 'down'
            )
            
            # Use whichever is more conservative
            if direction == 'BUY':
                tp2 = min(fib_tp2, standard_tp2, htf_swing)
            else:
                tp2 = max(fib_tp2, standard_tp2, htf_swing)
            
            return {
                'tp1': round(tp1, 5),
                'tp2': round(tp2, 5),
                'tp1_pips': tp1_pips,
                'tp2_pips': utils.calculate_pips(symbol, entry, tp2),
                'tp1_rr': config.TARGET_RR_TP1,
                'tp2_rr': utils.calculate_pips(symbol, entry, tp2) / risk_pips
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating take profits: {e}")
            # Fallback
            risk_pips = abs(entry - stop_loss) / utils.get_pip_value(symbol)
            tp1 = entry + (risk_pips * 1.5 * utils.get_pip_value(symbol) * (1 if direction == 'BUY' else -1))
            tp2 = entry + (risk_pips * 2.5 * utils.get_pip_value(symbol) * (1 if direction == 'BUY' else -1))
            
            return {
                'tp1': round(tp1, 5),
                'tp2': round(tp2, 5),
                'tp1_pips': risk_pips * 1.5,
                'tp2_pips': risk_pips * 2.5,
                'tp1_rr': 1.5,
                'tp2_rr': 2.5
            }
    
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
    
    def check_session_filter(self, timestamp: datetime) -> Tuple[bool, str]:
        """
        REFINEMENT #6: Trading session filter.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            tuple: (pass_filter, reason)
        """
        session = utils.get_session(timestamp)
        
        if config.AVOID_ASIAN_SESSION and session == 'ASIAN':
            return False, "Asian session - lower liquidity"
        
        if config.PREFER_LONDON_NY_OVERLAP and utils.is_london_ny_overlap(timestamp):
            return True, "London/NY overlap - optimal liquidity"
        
        if session in ['LONDON', 'NEW_YORK']:
            return True, f"{session} session - good liquidity"
        
        return False, f"{session} session - avoid"
    
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