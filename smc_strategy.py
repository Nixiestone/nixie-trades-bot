"""
smc_strategy.py - Smart Money Concepts Strategy Implementation
Nix Trades Telegram Bot

Implements SMC trading logic with 8 precision refinements:
1. Volume-weighted Order Block detection
2. Inducement quality validation (wick-to-body ratio)
3. ATR-adjusted dynamic stop loss
4. Session filtering (Asian/London/NY)
5. Fibonacci-based adaptive TP2
6. Currency exposure limits
7. Volatility regime filtering
8. Drawdown-based risk adjustment

NO EMOJIS - Professional code only
"""

import logging
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timezone, timedelta
import numpy as np
from decimal import Decimal
import config
import utils

logger = logging.getLogger(__name__)


class SMCStrategy:
    """
    Smart Money Concepts trading strategy with institutional-grade refinements.
    """
    
    def __init__(self):
        """Initialize SMC strategy with default parameters."""
        self.htf_lookback = 20  # Candles to analyze for trend
        self.volume_ma_period = 20  # Moving average period for volume
        self.atr_period = config.ATR_PERIOD
        
        logger.info("SMC Strategy initialized with 8 precision refinements")
    
    
    # ==================== HTF TREND ANALYSIS ====================
    
    def determine_htf_trend(self, candles: np.ndarray) -> str:
        """
        Determine Higher Timeframe trend direction.
        
        Args:
            candles: OHLCV data array (last 20+ candles from Daily/4H)
            
        Returns:
            str: 'BULLISH', 'BEARISH', or 'RANGING'
            
        Logic:
            Count Higher Highs vs Lower Lows
            If HH > LL by 60%+ -> BULLISH
            If LL > HH by 60%+ -> BEARISH
            Else -> RANGING
        """
        try:
            if len(candles) < self.htf_lookback:
                logger.warning("Insufficient candles for HTF trend analysis")
                return 'RANGING'
            
            highs = candles[-self.htf_lookback:, 1]  # High prices
            lows = candles[-self.htf_lookback:, 2]   # Low prices
            
            higher_highs = 0
            lower_lows = 0
            
            for i in range(1, len(highs)):
                if highs[i] > highs[i-1]:
                    higher_highs += 1
                if lows[i] < lows[i-1]:
                    lower_lows += 1
            
            total_swings = higher_highs + lower_lows
            
            if total_swings == 0:
                return 'RANGING'
            
            hh_ratio = higher_highs / total_swings
            ll_ratio = lower_lows / total_swings
            
            if hh_ratio >= 0.6:
                return 'BULLISH'
            elif ll_ratio >= 0.6:
                return 'BEARISH'
            else:
                return 'RANGING'
        
        except Exception as e:
            logger.error(f"Error in determine_htf_trend: {e}")
            return 'RANGING'
    
    
    # ==================== REFINEMENT 1: VOLUME-WEIGHTED ORDER BLOCK ====================
    
    def detect_order_block_with_volume(
        self,
        candles: np.ndarray,
        direction: str
    ) -> Optional[Dict[str, Any]]:
        """
        Detect Order Block with volume confirmation.
        
        Args:
            candles: OHLCV data (last 30+ candles from entry timeframe)
            direction: 'BUY' or 'SELL'
            
        Returns:
            dict: OB data with {high, low, body_high, body_low, volume_confirmed}
            or None if no valid OB
            
        Refinement 1 Logic:
            - OB candle must have >= 1.5x average volume
            - Impulse (next 5 candles) must have >= 2.0x average volume
        """
        try:
            if len(candles) < 30:
                return None
            
            # Calculate volume moving average
            volumes = candles[-30:, 4]  # Volume column
            avg_volume = np.mean(volumes)
            
            # Search for Order Block (last opposite candle before impulse)
            for i in range(len(candles) - 6, max(0, len(candles) - 20), -1):
                current_candle = candles[i]
                next_candles = candles[i+1:i+6]
                
                # Check if this is last opposite candle
                is_bullish = current_candle[3] > current_candle[0]  # Close > Open
                is_bearish = current_candle[3] < current_candle[0]  # Close < Open
                
                if direction == 'BUY' and not is_bearish:
                    continue
                if direction == 'SELL' and not is_bullish:
                    continue
                
                # Check volume confirmation
                ob_volume = current_candle[4]
                volume_confirmed = ob_volume >= (avg_volume * config.VOLUME_THRESHOLD_OB)
                
                if not volume_confirmed:
                    continue
                
                # Check impulse volume
                impulse_volumes = next_candles[:, 4]
                impulse_avg = np.mean(impulse_volumes)
                strong_impulse = impulse_avg >= (avg_volume * config.VOLUME_THRESHOLD_IMPULSE)
                
                if not strong_impulse:
                    continue
                
                # Valid OB found
                return {
                    'high': float(current_candle[1]),
                    'low': float(current_candle[2]),
                    'body_high': float(max(current_candle[0], current_candle[3])),
                    'body_low': float(min(current_candle[0], current_candle[3])),
                    'volume_confirmed': True,
                    'ob_volume_ratio': float(ob_volume / avg_volume),
                    'impulse_volume_ratio': float(impulse_avg / avg_volume)
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Error in detect_order_block_with_volume: {e}")
            return None
    
    
    # ==================== REFINEMENT 2: INDUCEMENT QUALITY ====================
    
    def validate_inducement_quality(
        self,
        inducement_candle: np.ndarray,
        liquidity_level: float,
        direction: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate inducement quality using wick-to-body ratio.
        
        Args:
            inducement_candle: OHLC data for inducement candle
            liquidity_level: Price level that was swept
            direction: 'BUY' or 'SELL'
            
        Returns:
            Tuple[bool, dict]: (is_valid, quality_metrics)
            
        Refinement 2 Logic:
            - Wick swept 3-10 pips beyond liquidity
            - Body did NOT close past liquidity (rejection)
            - Wick is >= 60% of total candle range
        """
        try:
            open_price = float(inducement_candle[0])
            high = float(inducement_candle[1])
            low = float(inducement_candle[2])
            close = float(inducement_candle[3])
            
            candle_range = high - low
            
            if candle_range == 0:
                return False, {}
            
            if direction == 'BUY':
                # For buy setup, check downward sweep
                sweep_distance = abs(low - liquidity_level)
                body_safe = close > liquidity_level
                wick_length = close - low  # Assuming bullish rejection
                wick_dominance = wick_length / candle_range
                
            else:  # SELL
                # For sell setup, check upward sweep
                sweep_distance = abs(high - liquidity_level)
                body_safe = close < liquidity_level
                wick_length = high - close  # Assuming bearish rejection
                wick_dominance = wick_length / candle_range
            
            # Convert sweep distance to pips (approximate for forex)
            sweep_pips = sweep_distance / 0.0001  # Standard pip size
            
            # Validate criteria
            swept_correctly = config.INDUCEMENT_MIN_PIPS <= sweep_pips <= config.INDUCEMENT_MAX_PIPS
            wick_strong = wick_dominance >= config.INDUCEMENT_WICK_RATIO
            
            is_valid = swept_correctly and body_safe and wick_strong
            
            quality_metrics = {
                'sweep_pips': float(sweep_pips),
                'body_safe': body_safe,
                'wick_dominance': float(wick_dominance),
                'valid': is_valid
            }
            
            return is_valid, quality_metrics
        
        except Exception as e:
            logger.error(f"Error in validate_inducement_quality: {e}")
            return False, {}
    
    
    # ==================== REFINEMENT 3: ATR-ADJUSTED STOP LOSS ====================
    
    def calculate_dynamic_stop_loss(
        self,
        entry_price: float,
        ob_level: float,
        atr_value: float,
        spread: float,
        direction: str
    ) -> Tuple[Optional[float], str]:
        """
        Calculate ATR-adjusted stop loss with spread protection.
        
        Args:
            entry_price: Entry price
            ob_level: Order Block low (for BUY) or high (for SELL)
            atr_value: Current ATR(14) value
            spread: Current broker spread in pips
            direction: 'BUY' or 'SELL'
            
        Returns:
            Tuple[float, str]: (stop_loss_price, explanation) or (None, error_msg)
            
        Refinement 3 Logic:
            SL = OB level - (spread_buffer + volatility_buffer + structural_padding)
            Reject if risk > 50 pips
        """
        try:
            # Calculate buffers
            spread_buffer = spread * 1.5 * 0.0001  # 1.5x spread in price
            volatility_buffer = atr_value * 0.5    # 0.5 ATR
            structural_padding = 3 * 0.0001        # 3 pips
            
            total_padding = spread_buffer + volatility_buffer + structural_padding
            
            if direction == 'BUY':
                stop_loss = ob_level - total_padding
            else:  # SELL
                stop_loss = ob_level + total_padding
            
            # Calculate risk in pips
            risk_pips = abs(entry_price - stop_loss) / 0.0001
            
            # Enforce maximum risk
            if risk_pips > config.MAX_RISK_PIPS:
                return None, f"Risk exceeds {config.MAX_RISK_PIPS} pips (calculated: {risk_pips:.1f})"
            
            explanation = (
                f"SL: {stop_loss:.5f}, Risk: {risk_pips:.1f} pips "
                f"(ATR: {atr_value:.5f}, Spread: {spread} pips)"
            )
            
            return stop_loss, explanation
        
        except Exception as e:
            logger.error(f"Error in calculate_dynamic_stop_loss: {e}")
            return None, str(e)
    
    
    # ==================== REFINEMENT 4: SESSION FILTERING ====================
    
    def should_trade_in_session(self, utc_hour: int, setup_quality: str) -> Tuple[bool, str]:
        """
        Determine if trading is allowed in current session.
        
        Args:
            utc_hour: Current hour in UTC (0-23)
            setup_quality: 'UNICORN' or 'STANDARD'
            
        Returns:
            Tuple[bool, str]: (allowed, reason)
            
        Refinement 4 Logic:
            - Asian (0-7 UTC): Only Unicorn setups
            - London Open (7-8 UTC): NO TRADING (spread widening)
            - London/NY/Overlap: All setups allowed
        """
        try:
            session_name = utils.get_session_name(utc_hour)
            
            if utc_hour >= 0 and utc_hour < 7:  # Asian
                if setup_quality == 'UNICORN':
                    return True, "Asian session - Unicorn setup approved"
                else:
                    return False, "Asian session - only Unicorn setups allowed"
            
            elif utc_hour == 7:  # London Open
                return False, "London open - spread widening, no trading"
            
            else:  # London, NY, Overlap
                return True, f"{session_name} session - all setups allowed"
        
        except Exception as e:
            logger.error(f"Error in should_trade_in_session: {e}")
            return False, str(e)
    
    
    # ==================== REFINEMENT 5: FIBONACCI TP2 ====================
    
    def calculate_adaptive_tp2(
        self,
        entry_price: float,
        tp1: float,
        htf_high: float,
        htf_low: float,
        trend_strength: str,
        stop_loss: float,
        direction: str
    ) -> float:
        """
        Calculate Fibonacci-based TP2 adjusted for trend strength.
        
        Args:
            entry_price: Entry price
            tp1: Take Profit 1 level
            htf_high: HTF range high
            htf_low: HTF range low
            trend_strength: 'STRONG', 'MODERATE', or 'WEAK'
            stop_loss: Stop loss price
            direction: 'BUY' or 'SELL'
            
        Returns:
            float: TP2 price
            
        Refinement 5 Logic:
            Strong trend: 100% of range
            Moderate trend: 61.8% Fibonacci
            Weak trend: 50% between TP1 and full range
        """
        try:
            range_size = abs(htf_high - htf_low)
            
            # Get Fibonacci level for trend strength
            fib_level = config.FIBONACCI_LEVELS.get(
                trend_strength.lower() + '_trend',
                0.618  # Default to moderate
            )
            
            if direction == 'BUY':
                fib_tp2 = entry_price + (range_size * fib_level)
            else:  # SELL
                fib_tp2 = entry_price - (range_size * fib_level)
            
            # Ensure minimum R:R of 2.5:1
            risk = abs(entry_price - stop_loss)
            minimum_tp2 = entry_price + (risk * 2.5) if direction == 'BUY' else entry_price - (risk * 2.5)
            
            if direction == 'BUY':
                tp2 = max(fib_tp2, minimum_tp2)
            else:
                tp2 = min(fib_tp2, minimum_tp2)
            
            return tp2
        
        except Exception as e:
            logger.error(f"Error in calculate_adaptive_tp2: {e}")
            # Fallback: Use simple 2.5:1 R:R
            risk = abs(entry_price - stop_loss)
            return entry_price + (risk * 2.5) if direction == 'BUY' else entry_price - (risk * 2.5)
    
    
    # ==================== REFINEMENT 6: CURRENCY EXPOSURE ====================
    
    def check_currency_exposure(
        self,
        open_trades: List[Dict[str, Any]],
        new_symbol: str,
        new_direction: str
    ) -> Tuple[bool, str]:
        """
        Check if adding new trade would exceed currency exposure limits.
        
        Args:
            open_trades: List of currently open trades
            new_symbol: Symbol for proposed trade
            new_direction: 'BUY' or 'SELL'
            
        Returns:
            Tuple[bool, str]: (allowed, reason)
            
        Refinement 6 Logic:
            No single currency exposure > 3 trades
            Prevents correlation risk
        """
        try:
            # Initialize exposure counter
            exposure = {
                'USD': 0, 'EUR': 0, 'GBP': 0, 'JPY': 0,
                'AUD': 0, 'CAD': 0, 'CHF': 0, 'NZD': 0,
                'XAU': 0, 'XAG': 0
            }
            
            # Count current exposure
            for trade in open_trades:
                symbol = trade['symbol']
                direction = trade['direction']
                
                base, quote = utils.extract_currency_from_symbol(symbol)
                
                if direction == 'BUY':
                    exposure[base] = exposure.get(base, 0) + 1
                    exposure[quote] = exposure.get(quote, 0) - 1
                else:  # SELL
                    exposure[base] = exposure.get(base, 0) - 1
                    exposure[quote] = exposure.get(quote, 0) + 1
            
            # Simulate adding new trade
            new_base, new_quote = utils.extract_currency_from_symbol(new_symbol)
            
            test_exposure = exposure.copy()
            if new_direction == 'BUY':
                test_exposure[new_base] = test_exposure.get(new_base, 0) + 1
                test_exposure[new_quote] = test_exposure.get(new_quote, 0) - 1
            else:
                test_exposure[new_base] = test_exposure.get(new_base, 0) - 1
                test_exposure[new_quote] = test_exposure.get(new_quote, 0) + 1
            
            # Check limits
            for currency, exp in test_exposure.items():
                if abs(exp) > config.MAX_CURRENCY_EXPOSURE:
                    return False, f"Currency exposure limit: {currency} would be {exp}"
            
            return True, "Currency exposure within limits"
        
        except Exception as e:
            logger.error(f"Error in check_currency_exposure: {e}")
            return True, "Exposure check failed - allowing trade (fail-safe)"
    
    
    # ==================== REFINEMENT 7: VOLATILITY REGIME ====================
    
    def check_volatility_regime(self, current_atr: float, atr_20_day_avg: float) -> Tuple[bool, str]:
        """
        Validate current volatility is within acceptable range.
        
        Args:
            current_atr: Current ATR(14) value
            atr_20_day_avg: 20-day average of ATR
            
        Returns:
            Tuple[bool, str]: (valid, reason)
            
        Refinement 7 Logic:
            Reject if ATR < 0.7x average (too quiet, choppy)
            Reject if ATR > 2.0x average (too volatile, stop-outs)
        """
        try:
            if atr_20_day_avg == 0:
                return False, "ATR average is zero - data issue"
            
            volatility_ratio = current_atr / atr_20_day_avg
            
            if volatility_ratio < config.ATR_MIN_RATIO:
                return False, f"Low volatility regime ({volatility_ratio:.2f}x) - skip trade"
            
            elif volatility_ratio > config.ATR_MAX_RATIO:
                return False, f"High volatility regime ({volatility_ratio:.2f}x) - skip trade"
            
            else:
                return True, f"Normal volatility regime ({volatility_ratio:.2f}x)"
        
        except Exception as e:
            logger.error(f"Error in check_volatility_regime: {e}")
            return False, str(e)
    
    
    # ==================== REFINEMENT 8: DRAWDOWN PROTECTION ====================
    
    def calculate_adaptive_risk(
        self,
        base_risk_percent: float,
        current_drawdown_percent: float
    ) -> Tuple[float, str]:
        """
        Adjust risk percentage based on current drawdown.
        
        Args:
            base_risk_percent: User's base risk setting (e.g., 1.0)
            current_drawdown_percent: Current account drawdown
            
        Returns:
            Tuple[float, str]: (adjusted_risk_percent, reason)
            
        Refinement 8 Logic:
            0-3% DD: Normal risk (1.0x)
            3-5% DD: Reduced risk (0.7x)
            5-8% DD: Conservative risk (0.5x)
            >8% DD: HALT trading (0x)
        """
        try:
            if current_drawdown_percent < config.DRAWDOWN_LEVEL_1:
                multiplier = config.RISK_MULTIPLIER_LEVEL_1
                reason = "Normal risk - no drawdown"
            
            elif current_drawdown_percent < config.DRAWDOWN_LEVEL_2:
                multiplier = config.RISK_MULTIPLIER_LEVEL_2
                reason = f"Reduced risk - {current_drawdown_percent:.1f}% drawdown"
            
            elif current_drawdown_percent < config.DRAWDOWN_LEVEL_3:
                multiplier = config.RISK_MULTIPLIER_LEVEL_3
                reason = f"Conservative risk - {current_drawdown_percent:.1f}% drawdown"
            
            else:
                multiplier = config.RISK_MULTIPLIER_HALT
                reason = f"TRADING HALTED - {current_drawdown_percent:.1f}% drawdown (>8% limit)"
            
            adjusted_risk = base_risk_percent * multiplier
            
            return adjusted_risk, reason
        
        except Exception as e:
            logger.error(f"Error in calculate_adaptive_risk: {e}")
            return base_risk_percent, "Error calculating - using base risk"
    
    
    # ==================== 12-POINT VALIDATION CHECKLIST ====================
    
    def validate_complete_setup(
        self,
        htf_trend: str,
        expected_direction: str,
        setup_type: str,
        bos_count: int,
        ob_data: Dict[str, Any],
        inducement_valid: bool,
        session_ok: bool,
        volatility_ok: bool,
        exposure_ok: bool,
        news_proximity_minutes: int,
        rr_ratio: float,
        ml_score: int,
        current_drawdown: float,
        setup_quality: str
    ) -> Tuple[bool, str, int]:
        """
        12-point validation checklist before approving setup.
        
        Returns:
            Tuple[bool, str, int]: (approved, reason, confidence_score)
        """
        try:
            checks_passed = 0
            total_checks = 12
            confidence_score = 0
            
            # 1. HTF Trend Alignment
            if htf_trend == expected_direction or setup_type == 'MSS_REVERSAL':
                checks_passed += 1
                confidence_score += 10
            else:
                return False, "HTF trend misalignment", 0
            
            # 2. Double BOS (for continuations)
            if setup_type == 'BOS_CONTINUATION':
                if bos_count >= 2:
                    checks_passed += 1
                    confidence_score += 15
                else:
                    return False, "Double BOS not confirmed", 0
            else:
                checks_passed += 1
            
            # 3. Volume Confirmation
            if ob_data.get('volume_confirmed'):
                checks_passed += 1
                confidence_score += 12
            else:
                return False, "Insufficient volume confirmation", 0
            
            # 4. Inducement Quality
            if inducement_valid:
                checks_passed += 1
                confidence_score += 10
            else:
                return False, "Poor inducement quality", 0
            
            # 5. Session Filter
            if session_ok:
                checks_passed += 1
                confidence_score += 8
            else:
                return False, "Session filter blocked trade", 0
            
            # 6. Volatility Regime
            if volatility_ok:
                checks_passed += 1
                confidence_score += 8
            else:
                return False, "Volatility regime unfavorable", 0
            
            # 7. Currency Exposure
            if exposure_ok:
                checks_passed += 1
                confidence_score += 7
            else:
                return False, "Currency exposure limit exceeded", 0
            
            # 8. News Proximity
            if news_proximity_minutes > 30 or news_proximity_minutes == -1:
                checks_passed += 1
                confidence_score += 5
            else:
                return False, f"News in {news_proximity_minutes}m - too close", 0
            
            # 9. Risk-Reward
            if rr_ratio >= 1.5:
                checks_passed += 1
                confidence_score += 10
            else:
                return False, f"R:R too low ({rr_ratio:.2f})", 0
            
            # 10. ML Agreement
            if ml_score >= config.ML_THRESHOLD:
                checks_passed += 1
                confidence_score += ml_score // 10
            else:
                return False, f"ML score too low ({ml_score}%)", 0
            
            # 11. Drawdown Protection
            if current_drawdown < 8.0:
                checks_passed += 1
                confidence_score += 5
            else:
                return False, "Drawdown >8% - trading halted", 0
            
            # 12. Setup Quality Bonus
            if setup_quality == 'UNICORN':
                checks_passed += 1
                confidence_score += 10
            else:
                checks_passed += 1
            
            # All checks passed
            if setup_quality == 'UNICORN':
                confidence_score = min(confidence_score, 95)
            else:
                confidence_score = min(confidence_score, 85)
            
            return True, "All validation checks passed", confidence_score
        
        except Exception as e:
            logger.error(f"Error in validate_complete_setup: {e}")
            return False, f"Validation error: {str(e)}", 0