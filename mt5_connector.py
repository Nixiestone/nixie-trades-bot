import MetaTrader5 as mt5
import logging
from typing import Optional, Dict, Tuple, List, Any
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_DOWN
import config
import utils

logger = logging.getLogger(__name__)


# ==================== CONNECTION MANAGEMENT ====================

def initialize_mt5() -> bool:
    """
    Initialize MetaTrader 5 connection.
    
    Returns:
        bool: True if successful, False otherwise
        
    Note:
        Must be called before any MT5 operations
    """
    try:
        if not mt5.initialize():
            error_code = mt5.last_error()
            logger.error(f"MT5 initialization failed: {error_code}")
            return False
        
        logger.info("MT5 initialized successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error initializing MT5: {e}")
        return False


def shutdown_mt5() -> None:
    """Shutdown MetaTrader 5 connection cleanly."""
    try:
        mt5.shutdown()
        logger.info("MT5 shutdown successfully")
    except Exception as e:
        logger.error(f"Error shutting down MT5: {e}")


def login_mt5(login: str, password: str, server: str) -> bool:
    """
    Login to MT5 account.
    
    Args:
        login: MT5 account login number
        password: MT5 account password
        server: MT5 server name
        
    Returns:
        bool: True if login successful
    """
    try:
        # Ensure MT5 is initialized
        if not mt5.terminal_info():
            if not initialize_mt5():
                return False
        
        # Attempt login
        authorized = mt5.login(login=int(login), password=password, server=server)
        
        if not authorized:
            error_code = mt5.last_error()
            logger.error(f"MT5 login failed: {error_code}")
            return False
        
        # Verify account info
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to retrieve account info after login")
            return False
        
        logger.info(f"MT5 login successful: {login}@{server}, Balance: {account_info.balance}")
        return True
    
    except Exception as e:
        logger.error(f"Error during MT5 login: {e}")
        return False


def test_connection() -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Test MT5 connection and retrieve account info.
    
    Returns:
        Tuple[bool, dict]: (success, account_info_dict)
    """
    try:
        account_info = mt5.account_info()
        
        if account_info is None:
            return False, None
        
        info_dict = {
            'login': account_info.login,
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'leverage': account_info.leverage,
            'currency': account_info.currency,
            'server': account_info.server,
            'company': account_info.company
        }
        
        return True, info_dict
    
    except Exception as e:
        logger.error(f"Error testing MT5 connection: {e}")
        return False, None


# ==================== SYMBOL NORMALIZATION ====================

def normalize_symbol(standard_symbol: str, user_symbol_mappings: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Convert standard symbol to broker-specific symbol format.
    
    Args:
        standard_symbol: Standard symbol (e.g., 'EURUSD', 'XAUUSD')
        user_symbol_mappings: User's saved symbol mappings
        
    Returns:
        str: Broker symbol if found, None if not available
        
    Process:
        1. Check user's saved mappings
        2. Try exact match
        3. Try common variations (.pro, .raw, -a, m suffix)
        4. Try fuzzy match
        5. Return None if not found (prompt user to provide mapping)
    """
    try:
        # Step 1: Check user's saved mappings
        if user_symbol_mappings and standard_symbol in user_symbol_mappings:
            broker_symbol = user_symbol_mappings[standard_symbol]
            
            # Verify it exists on broker
            symbol_info = mt5.symbol_info(broker_symbol)
            if symbol_info is not None:
                logger.info(f"Symbol normalization: {standard_symbol} -> {broker_symbol} (user mapping)")
                return broker_symbol
        
        # Step 2: Try exact match
        symbol_info = mt5.symbol_info(standard_symbol)
        if symbol_info is not None:
            logger.info(f"Symbol normalization: {standard_symbol} -> {standard_symbol} (exact match)")
            return standard_symbol
        
        # Step 3: Try common variations
        if standard_symbol in config.SYMBOL_VARIATIONS:
            variations = config.SYMBOL_VARIATIONS[standard_symbol]
            
            for variation in variations:
                symbol_info = mt5.symbol_info(variation)
                if symbol_info is not None:
                    logger.info(f"Symbol normalization: {standard_symbol} -> {variation} (variation match)")
                    return variation
        
        # Step 4: Try fuzzy match (search all available symbols)
        all_symbols = mt5.symbols_get()
        if all_symbols:
            # Remove common suffixes for comparison
            clean_standard = standard_symbol.replace('USD', '').replace('XAU', 'GOLD').replace('XAG', 'SILVER')
            
            for symbol_obj in all_symbols:
                symbol_name = symbol_obj.name
                
                # Check if standard symbol is contained in broker symbol
                if standard_symbol.upper() in symbol_name.upper():
                    logger.info(f"Symbol normalization: {standard_symbol} -> {symbol_name} (fuzzy match)")
                    return symbol_name
        
        # Step 5: Not found
        logger.warning(f"Symbol normalization failed: {standard_symbol} not found on broker")
        return None
    
    except Exception as e:
        logger.error(f"Error normalizing symbol {standard_symbol}: {e}")
        return None


def get_symbol_info(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed symbol information.
    
    Args:
        symbol: Broker-specific symbol
        
    Returns:
        dict: Symbol info including pip size, lot sizes, spread
    """
    try:
        symbol_info = mt5.symbol_info(symbol)
        
        if symbol_info is None:
            return None
        
        # Enable symbol for trading if not enabled
        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)
        
        info_dict = {
            'symbol': symbol,
            'digits': symbol_info.digits,
            'point': symbol_info.point,
            'pip_size': symbol_info.point * (10 if symbol_info.digits in [3, 5] else 1),
            'spread': symbol_info.spread,
            'volume_min': symbol_info.volume_min,
            'volume_max': symbol_info.volume_max,
            'volume_step': symbol_info.volume_step,
            'trade_contract_size': symbol_info.trade_contract_size,
            'currency_base': symbol_info.currency_base,
            'currency_profit': symbol_info.currency_profit
        }
        
        return info_dict
    
    except Exception as e:
        logger.error(f"Error getting symbol info for {symbol}: {e}")
        return None


# ==================== ORDER TYPE DETECTION ====================

def determine_order_type(
    symbol: str,
    direction: str,
    entry_price: float
) -> Dict[str, Any]:
    """
    CRITICAL FUNCTION: Determine order type based on distance to current price.
    This information is MANDATORY in every setup alert.
    
    Args:
        symbol: Trading symbol
        direction: 'BUY' or 'SELL'
        entry_price: Desired entry price
        
    Returns:
        dict: {
            'order_type': 'MARKET' | 'LIMIT' | 'STOP',
            'current_price': float,
            'distance_pips': float,
            'explanation': str (user-friendly explanation for Telegram message),
            'expiry_minutes': int (0 for market, 60 for limit/stop)
        }
        
    Logic:
        - Market Order: Entry within 2 pips of current price (execute immediately)
        - Limit Order: Entry 3-20 pips away (wait for pullback, expires in 1 hour)
        - Stop Order: Entry >20 pips away (breakout entry, expires in 1 hour)
    """
    try:
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick data for {symbol}")
            return {
                'order_type': 'UNKNOWN',
                'current_price': 0.0,
                'distance_pips': 0.0,
                'explanation': 'Unable to determine order type - market data unavailable',
                'expiry_minutes': 0
            }
        
        # Get pip size
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return {
                'order_type': 'UNKNOWN',
                'current_price': 0.0,
                'distance_pips': 0.0,
                'explanation': 'Unable to determine order type - symbol info unavailable',
                'expiry_minutes': 0
            }
        
        pip_size = symbol_info['pip_size']
        
        # Determine relevant current price based on direction
        if direction.upper() == 'BUY':
            current_price = tick.ask
            price_diff = entry_price - current_price
        else:  # SELL
            current_price = tick.bid
            price_diff = current_price - entry_price
        
        # Calculate distance in pips
        distance_pips = abs(price_diff) / pip_size
        
        # Determine order type based on distance
        if distance_pips <= config.MARKET_ORDER_THRESHOLD_PIPS:
            # Market Order: Within 2 pips
            order_type = 'MARKET'
            explanation = (
                f"MARKET ORDER - Entry within {distance_pips:.1f} pips of current price.\n"
                f"Current {direction} price: {current_price:.5f}\n"
                f"Entry price: {entry_price:.5f}\n"
                f"This trade will execute IMMEDIATELY at the best available market price."
            )
            expiry_minutes = 0
        
        elif distance_pips <= config.LIMIT_ORDER_THRESHOLD_PIPS:
            # Limit Order: 3-20 pips away
            order_type = 'LIMIT'
            
            if direction.upper() == 'BUY':
                explanation = (
                    f"LIMIT ORDER - Entry {distance_pips:.1f} pips BELOW current price.\n"
                    f"Current ASK price: {current_price:.5f}\n"
                    f"Entry price: {entry_price:.5f}\n"
                    f"Waiting for price to pull back to entry level.\n"
                    f"Order expires in {config.LIMIT_ORDER_EXPIRY_MINUTES} minutes if not filled."
                )
            else:  # SELL
                explanation = (
                    f"LIMIT ORDER - Entry {distance_pips:.1f} pips ABOVE current price.\n"
                    f"Current BID price: {current_price:.5f}\n"
                    f"Entry price: {entry_price:.5f}\n"
                    f"Waiting for price to pull back to entry level.\n"
                    f"Order expires in {config.LIMIT_ORDER_EXPIRY_MINUTES} minutes if not filled."
                )
            
            expiry_minutes = config.LIMIT_ORDER_EXPIRY_MINUTES
        
        else:
            # Stop Order: >20 pips away
            order_type = 'STOP'
            
            if direction.upper() == 'BUY':
                explanation = (
                    f"STOP ORDER - Entry {distance_pips:.1f} pips ABOVE current price.\n"
                    f"Current ASK price: {current_price:.5f}\n"
                    f"Entry price: {entry_price:.5f}\n"
                    f"Waiting for breakout above entry level.\n"
                    f"Order expires in {config.STOP_ORDER_EXPIRY_MINUTES} minutes if not filled."
                )
            else:  # SELL
                explanation = (
                    f"STOP ORDER - Entry {distance_pips:.1f} pips BELOW current price.\n"
                    f"Current BID price: {current_price:.5f}\n"
                    f"Entry price: {entry_price:.5f}\n"
                    f"Waiting for breakout below entry level.\n"
                    f"Order expires in {config.STOP_ORDER_EXPIRY_MINUTES} minutes if not filled."
                )
            
            expiry_minutes = config.STOP_ORDER_EXPIRY_MINUTES
        
        result = {
            'order_type': order_type,
            'current_price': current_price,
            'distance_pips': distance_pips,
            'explanation': explanation,
            'expiry_minutes': expiry_minutes
        }
        
        logger.info(f"Order type determination: {symbol} {direction} - {order_type} ({distance_pips:.1f} pips)")
        
        return result
    
    except Exception as e:
        logger.error(f"Error determining order type: {e}")
        return {
            'order_type': 'UNKNOWN',
            'current_price': 0.0,
            'distance_pips': 0.0,
            'explanation': f'Error determining order type: {str(e)}',
            'expiry_minutes': 0
        }


# ==================== POSITION SIZING ====================

def calculate_lot_size(
    symbol: str,
    account_balance: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float
) -> Tuple[Optional[float], Optional[str]]:
    """
    Calculate lot size based on risk percentage.
    
    Args:
        symbol: Trading symbol
        account_balance: Account balance in account currency
        risk_percent: Risk percentage per trade (e.g., 1.0 for 1%)
        entry_price: Entry price
        stop_loss: Stop loss price
        
    Returns:
        Tuple[float, str]: (lot_size, error_message)
        Returns (None, error_msg) if calculation fails
        
    Formula:
        lot_size = (account_balance * risk_percent / 100) / (sl_pips * pip_value)
    """
    try:
        # Get symbol info
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return None, f"Symbol {symbol} not available"
        
        pip_size = symbol_info['pip_size']
        
        # Calculate SL distance in pips
        sl_pips = abs(entry_price - stop_loss) / pip_size
        
        if sl_pips == 0:
            return None, "Stop loss distance is zero"
        
        if sl_pips > config.MAX_RISK_PIPS:
            return None, f"Stop loss too wide ({sl_pips:.1f} pips > {config.MAX_RISK_PIPS} max)"
        
        # Calculate risk amount in account currency
        risk_amount = account_balance * (risk_percent / 100)
        
        # Get pip value (simplified calculation)
        # For forex pairs where quote currency = account currency (e.g., EURUSD account in USD)
        # pip_value = lot_size * contract_size * pip_size
        # Rearranged: lot_size = risk_amount / (sl_pips * pip_value_per_lot)
        
        contract_size = symbol_info['trade_contract_size']
        
        # Approximate pip value per standard lot
        # For JPY pairs: different calculation
        if 'JPY' in symbol:
            pip_value_per_lot = (pip_size * contract_size) / entry_price
        else:
            pip_value_per_lot = pip_size * contract_size
        
        # Calculate lot size
        lot_size = risk_amount / (sl_pips * pip_value_per_lot)
        
        # Apply limits and rounding
        lot_size = max(lot_size, symbol_info['volume_min'])
        lot_size = min(lot_size, symbol_info['volume_max'])
        
        # Round to valid step
        volume_step = symbol_info['volume_step']
        lot_size = round(lot_size / volume_step) * volume_step
        
        # Additional safety checks
        if lot_size < config.MIN_LOT_SIZE:
            lot_size = config.MIN_LOT_SIZE
        
        if lot_size > config.MAX_LOT_SIZE:
            return None, f"Calculated lot size ({lot_size:.2f}) exceeds maximum ({config.MAX_LOT_SIZE})"
        
        # Round to 2 decimals
        lot_size = round(lot_size, 2)
        
        logger.info(
            f"Lot size calculated: {symbol}, Balance: {account_balance}, "
            f"Risk: {risk_percent}%, SL: {sl_pips:.1f} pips, Lot: {lot_size}"
        )
        
        return lot_size, None
    
    except Exception as e:
        logger.error(f"Error calculating lot size: {e}")
        return None, f"Lot size calculation error: {str(e)}"


def check_margin_requirement(symbol: str, lot_size: float, order_type: str) -> Tuple[bool, Optional[str]]:
    """
    Check if account has sufficient margin for the trade.
    
    Args:
        symbol: Trading symbol
        lot_size: Lot size
        order_type: 'BUY' or 'SELL'
        
    Returns:
        Tuple[bool, str]: (sufficient, error_message)
    """
    try:
        account_info = mt5.account_info()
        if account_info is None:
            return False, "Account info unavailable"
        
        # Calculate required margin
        if order_type.upper() == 'BUY':
            action = mt5.ORDER_TYPE_BUY
        else:
            action = mt5.ORDER_TYPE_SELL
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False, f"Price data unavailable for {symbol}"
        
        price = tick.ask if order_type.upper() == 'BUY' else tick.bid
        
        # Calculate margin (simplified)
        margin_required = mt5.order_calc_margin(action, symbol, lot_size, price)
        
        if margin_required is None:
            return False, "Margin calculation failed"
        
        free_margin = account_info.margin_free
        
        if free_margin < margin_required:
            return False, f"Insufficient margin. Required: {margin_required:.2f}, Available: {free_margin:.2f}"
        
        logger.info(f"Margin check passed: Required {margin_required:.2f}, Available {free_margin:.2f}")
        return True, None
    
    except Exception as e:
        logger.error(f"Error checking margin: {e}")
        return False, f"Margin check error: {str(e)}"


# ==================== ORDER EXECUTION ====================

def place_order(
    symbol: str,
    direction: str,
    lot_size: float,
    entry_price: float,
    stop_loss: float,
    take_profit_1: float,
    take_profit_2: Optional[float] = None,
    order_type: str = 'MARKET',
    expiry_minutes: int = 60,
    magic_number: int = 234000
) -> Tuple[Optional[int], Optional[str]]:
    """
    Place trade order on MT5.
    
    Args:
        symbol: Trading symbol
        direction: 'BUY' or 'SELL'
        lot_size: Lot size
        entry_price: Entry price (for limit/stop orders)
        stop_loss: Stop loss price
        take_profit_1: First take profit price
        take_profit_2: Second take profit price (optional, for TP2)
        order_type: 'MARKET', 'LIMIT', or 'STOP'
        expiry_minutes: Order expiry time in minutes (for pending orders)
        magic_number: Magic number for order identification
        
    Returns:
        Tuple[int, str]: (order_ticket, error_message)
        Returns (ticket_number, None) if successful
        Returns (None, error_msg) if failed
        
    Note:
        For market orders, uses TP1 as primary take profit
        For managing TP1/TP2 split, use modify_position() after execution
    """
    try:
        # Determine MT5 order type
        if order_type.upper() == 'MARKET':
            if direction.upper() == 'BUY':
                mt5_order_type = mt5.ORDER_TYPE_BUY
            else:
                mt5_order_type = mt5.ORDER_TYPE_SELL
            price = 0.0  # Market execution uses current price
        
        elif order_type.upper() == 'LIMIT':
            if direction.upper() == 'BUY':
                mt5_order_type = mt5.ORDER_TYPE_BUY_LIMIT
            else:
                mt5_order_type = mt5.ORDER_TYPE_SELL_LIMIT
            price = entry_price
        
        else:  # STOP
            if direction.upper() == 'BUY':
                mt5_order_type = mt5.ORDER_TYPE_BUY_STOP
            else:
                mt5_order_type = mt5.ORDER_TYPE_SELL_STOP
            price = entry_price
        
        # Calculate expiry time for pending orders
        if order_type.upper() != 'MARKET':
            expiry_time = datetime.now(timezone.utc) + timedelta(minutes=expiry_minutes)
            expiry_timestamp = int(expiry_time.timestamp())
        else:
            expiry_timestamp = 0
        
        # Prepare request
        request = {
            'action': mt5.TRADE_ACTION_DEAL if order_type.upper() == 'MARKET' else mt5.TRADE_ACTION_PENDING,
            'symbol': symbol,
            'volume': lot_size,
            'type': mt5_order_type,
            'price': price,
            'sl': stop_loss,
            'tp': take_profit_1,  # Use TP1 as primary
            'deviation': 20,  # Max slippage in points
            'magic': magic_number,
            'comment': 'Nix Trades Bot',
            'type_time': mt5.ORDER_TIME_SPECIFIED if order_type.upper() != 'MARKET' else mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC
        }
        
        # Add expiry for pending orders
        if order_type.upper() != 'MARKET':
            request['expiration'] = expiry_timestamp
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            error_code = mt5.last_error()
            return None, f"Order send failed: {error_code}"
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = f"Order rejected: {result.comment} (retcode: {result.retcode})"
            logger.error(error_msg)
            return None, error_msg
        
        # Success
        ticket = result.order if order_type.upper() != 'MARKET' else result.deal
        
        logger.info(
            f"Order placed successfully: Ticket {ticket}, {symbol} {direction} "
            f"{lot_size} lots at {entry_price if order_type != 'MARKET' else result.price}"
        )
        
        return ticket, None
    
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return None, f"Order placement error: {str(e)}"


# ==================== POSITION MANAGEMENT ====================

def get_open_positions() -> List[Dict[str, Any]]:
    """
    Get all open positions.
    
    Returns:
        List[dict]: List of open position dictionaries
    """
    try:
        positions = mt5.positions_get()
        
        if positions is None:
            return []
        
        position_list = []
        
        for pos in positions:
            position_dict = {
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'volume': pos.volume,
                'open_price': pos.price_open,
                'current_price': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'swap': pos.swap,
                'commission': pos.commission if hasattr(pos, 'commission') else 0.0,
                'time': datetime.fromtimestamp(pos.time, tz=timezone.utc),
                'magic': pos.magic,
                'comment': pos.comment
            }
            
            position_list.append(position_dict)
        
        return position_list
    
    except Exception as e:
        logger.error(f"Error getting open positions: {e}")
        return []


def get_position_by_ticket(ticket: int) -> Optional[Dict[str, Any]]:
    """
    Get specific position by ticket number.
    
    Args:
        ticket: Position ticket number
        
    Returns:
        dict: Position dictionary, or None if not found
    """
    try:
        positions = mt5.positions_get(ticket=ticket)
        
        if positions is None or len(positions) == 0:
            return None
        
        pos = positions[0]
        
        position_dict = {
            'ticket': pos.ticket,
            'symbol': pos.symbol,
            'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
            'volume': pos.volume,
            'open_price': pos.price_open,
            'current_price': pos.price_current,
            'sl': pos.sl,
            'tp': pos.tp,
            'profit': pos.profit,
            'swap': pos.swap,
            'commission': pos.commission if hasattr(pos, 'commission') else 0.0,
            'time': datetime.fromtimestamp(pos.time, tz=timezone.utc),
            'magic': pos.magic,
            'comment': pos.comment
        }
        
        return position_dict
    
    except Exception as e:
        logger.error(f"Error getting position {ticket}: {e}")
        return None


def modify_position(
    ticket: int,
    new_sl: Optional[float] = None,
    new_tp: Optional[float] = None
) -> Tuple[bool, Optional[str]]:
    """
    Modify position SL/TP levels.
    
    Args:
        ticket: Position ticket number
        new_sl: New stop loss price (None to keep current)
        new_tp: New take profit price (None to keep current)
        
    Returns:
        Tuple[bool, str]: (success, error_message)
    """
    try:
        position = get_position_by_ticket(ticket)
        
        if position is None:
            return False, f"Position {ticket} not found"
        
        symbol = position['symbol']
        
        # Use current values if not changing
        sl = new_sl if new_sl is not None else position['sl']
        tp = new_tp if new_tp is not None else position['tp']
        
        request = {
            'action': mt5.TRADE_ACTION_SLTP,
            'position': ticket,
            'symbol': symbol,
            'sl': sl,
            'tp': tp
        }
        
        result = mt5.order_send(request)
        
        if result is None:
            error_code = mt5.last_error()
            return False, f"Modification failed: {error_code}"
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Modification rejected: {result.comment}"
        
        logger.info(f"Position {ticket} modified: SL={sl}, TP={tp}")
        return True, None
    
    except Exception as e:
        logger.error(f"Error modifying position {ticket}: {e}")
        return False, f"Modification error: {str(e)}"


def close_position(
    ticket: int,
    volume: Optional[float] = None
) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Close position (full or partial).
    
    Args:
        ticket: Position ticket number
        volume: Volume to close (None = close all)
        
    Returns:
        Tuple[bool, str, float]: (success, error_message, realized_profit)
    """
    try:
        position = get_position_by_ticket(ticket)
        
        if position is None:
            return False, f"Position {ticket} not found", None
        
        symbol = position['symbol']
        position_type = position['type']
        position_volume = position['volume']
        
        # Determine volume to close
        close_volume = volume if volume is not None else position_volume
        
        # Validate volume
        if close_volume > position_volume:
            return False, f"Close volume ({close_volume}) exceeds position volume ({position_volume})", None
        
        # Determine close order type
        if position_type == 'BUY':
            close_type = mt5.ORDER_TYPE_SELL
        else:
            close_type = mt5.ORDER_TYPE_BUY
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False, f"Price data unavailable for {symbol}", None
        
        close_price = tick.bid if position_type == 'BUY' else tick.ask
        
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'position': ticket,
            'symbol': symbol,
            'volume': close_volume,
            'type': close_type,
            'price': close_price,
            'deviation': 20,
            'comment': 'Nix Trades Bot Close'
        }
        
        result = mt5.order_send(request)
        
        if result is None:
            error_code = mt5.last_error()
            return False, f"Close failed: {error_code}", None
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Close rejected: {result.comment}", None
        
        # Calculate realized profit (approximate)
        realized_profit = result.profit if hasattr(result, 'profit') else None
        
        logger.info(f"Position {ticket} closed: Volume {close_volume}, Profit {realized_profit}")
        return True, None, realized_profit
    
    except Exception as e:
        logger.error(f"Error closing position {ticket}: {e}")
        return False, f"Close error: {str(e)}", None


def move_to_breakeven(ticket: int, buffer_pips: float = 5.0) -> Tuple[bool, Optional[str]]:
    """
    Move stop loss to breakeven + buffer.
    
    Args:
        ticket: Position ticket number
        buffer_pips: Buffer in pips above/below entry (default: 5 pips)
        
    Returns:
        Tuple[bool, str]: (success, error_message)
    """
    try:
        position = get_position_by_ticket(ticket)
        
        if position is None:
            return False, f"Position {ticket} not found"
        
        symbol = position['symbol']
        position_type = position['type']
        open_price = position['open_price']
        
        # Get pip size
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return False, f"Symbol info unavailable for {symbol}"
        
        pip_size = symbol_info['pip_size']
        
        # Calculate breakeven + buffer
        if position_type == 'BUY':
            new_sl = open_price + (buffer_pips * pip_size)
        else:
            new_sl = open_price - (buffer_pips * pip_size)
        
        # Modify position
        success, error = modify_position(ticket, new_sl=new_sl)
        
        if success:
            logger.info(f"Position {ticket} moved to breakeven + {buffer_pips} pips: SL={new_sl}")
        
        return success, error
    
    except Exception as e:
        logger.error(f"Error moving to breakeven: {e}")
        return False, f"Breakeven error: {str(e)}"


def check_pending_orders() -> List[Dict[str, Any]]:
    """
    Get all pending orders.
    
    Returns:
        List[dict]: List of pending order dictionaries
    """
    try:
        orders = mt5.orders_get()
        
        if orders is None:
            return []
        
        order_list = []
        
        for order in orders:
            order_dict = {
                'ticket': order.ticket,
                'symbol': order.symbol,
                'type': 'BUY' if 'BUY' in str(order.type) else 'SELL',
                'order_type': 'LIMIT' if 'LIMIT' in str(order.type) else 'STOP',
                'volume': order.volume_current,
                'price': order.price_open,
                'sl': order.sl,
                'tp': order.tp,
                'time_setup': datetime.fromtimestamp(order.time_setup, tz=timezone.utc),
                'expiration': datetime.fromtimestamp(order.time_expiration, tz=timezone.utc) if order.time_expiration > 0 else None,
                'magic': order.magic,
                'comment': order.comment
            }
            
            order_list.append(order_dict)
        
        return order_list
    
    except Exception as e:
        logger.error(f"Error getting pending orders: {e}")
        return []


def cancel_pending_order(ticket: int) -> Tuple[bool, Optional[str]]:
    """
    Cancel pending order.
    
    Args:
        ticket: Order ticket number
        
    Returns:
        Tuple[bool, str]: (success, error_message)
    """
    try:
        request = {
            'action': mt5.TRADE_ACTION_REMOVE,
            'order': ticket
        }
        
        result = mt5.order_send(request)
        
        if result is None:
            error_code = mt5.last_error()
            return False, f"Cancellation failed: {error_code}"
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Cancellation rejected: {result.comment}"
        
        logger.info(f"Pending order {ticket} cancelled")
        return True, None
    
    except Exception as e:
        logger.error(f"Error cancelling order {ticket}: {e}")
        return False, f"Cancellation error: {str(e)}"