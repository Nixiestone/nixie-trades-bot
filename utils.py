"""
NIX TRADES Utility Functions
Helper functions for time, calculations, formatting
NO EMOJIS - Professional code only
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, List
import pytz
import re
import config

logger = logging.getLogger(__name__)


# ==================== TIME UTILITIES ====================

def get_current_utc_time() -> datetime:
    """
    Get current UTC time with timezone awareness.
    
    Returns:
        datetime: Current UTC time
    """
    return datetime.now(timezone.utc)


def convert_utc_to_local(utc_time: datetime, user_timezone: str) -> datetime:
    """
    Convert UTC time to user's local timezone.
    
    Args:
        utc_time: UTC datetime object
        user_timezone: IANA timezone string (e.g., 'America/New_York')
        
    Returns:
        datetime: Local time in user's timezone
    """
    try:
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=timezone.utc)
        
        user_tz = pytz.timezone(user_timezone)
        local_time = utc_time.astimezone(user_tz)
        
        return local_time
    
    except Exception as e:
        logger.error(f"Error converting timezone: {e}")
        return utc_time


def format_dual_time(utc_time: datetime, user_timezone: str = 'UTC') -> str:
    """
    Format time as dual display: UTC and user's local time.
    
    Args:
        utc_time: UTC datetime object
        user_timezone: User's timezone
        
    Returns:
        str: Formatted string like "14:30 UTC (09:30 AM your time)"
    """
    try:
        utc_str = utc_time.strftime('%H:%M UTC')
        
        if user_timezone == 'UTC':
            return utc_str
        
        local_time = convert_utc_to_local(utc_time, user_timezone)
        local_str = local_time.strftime('%I:%M %p')
        
        return f"{utc_str} ({local_str} your time)"
    
    except Exception as e:
        logger.error(f"Error formatting dual time: {e}")
        return utc_time.strftime('%H:%M UTC')


def get_session(time_utc: Optional[datetime] = None) -> str:
    """
    Determine which trading session is active.
    
    Args:
        time_utc: UTC datetime (defaults to now)
        
    Returns:
        str: Session name or 'CLOSED'
    """
    if time_utc is None:
        time_utc = get_current_utc_time()
    
    hour = time_utc.hour
    minute = time_utc.minute
    current_minutes = hour * 60 + minute
    
    # Convert session times to minutes
    for session_name, times in config.SESSIONS.items():
        start_h, start_m = map(int, times['start'].split(':'))
        end_h, end_m = map(int, times['end'].split(':'))
        
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        
        if start_minutes <= current_minutes < end_minutes:
            return session_name
    
    return 'CLOSED'


def is_london_ny_overlap(time_utc: Optional[datetime] = None) -> bool:
    """
    Check if current time is during London/New York overlap.
    
    Args:
        time_utc: UTC datetime (defaults to now)
        
    Returns:
        bool: True if during overlap (13:00-17:00 UTC)
    """
    if time_utc is None:
        time_utc = get_current_utc_time()
    
    hour = time_utc.hour
    return 13 <= hour < 17


def calculate_time_until(target_time: datetime) -> str:
    """
    Calculate human-readable time until target.
    
    Args:
        target_time: Future datetime
        
    Returns:
        str: Formatted string like "1h 52m" or "45s"
    """
    now = get_current_utc_time()
    
    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=timezone.utc)
    
    delta = target_time - now
    
    if delta.total_seconds() < 0:
        return "0s"
    
    hours = int(delta.total_seconds() // 3600)
    minutes = int((delta.total_seconds() % 3600) // 60)
    seconds = int(delta.total_seconds() % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def calculate_trade_duration(entry_time: datetime, exit_time: Optional[datetime] = None) -> str:
    """
    Calculate how long a trade was open.
    
    Args:
        entry_time: When trade was opened
        exit_time: When trade was closed (defaults to now)
        
    Returns:
        str: Duration like "2h 15m" or "3d 4h"
    """
    if exit_time is None:
        exit_time = get_current_utc_time()
    
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=timezone.utc)
    if exit_time.tzinfo is None:
        exit_time = exit_time.replace(tzinfo=timezone.utc)
    
    delta = exit_time - entry_time
    
    days = delta.days
    hours = int(delta.seconds // 3600)
    minutes = int((delta.seconds % 3600) // 60)
    
    if days > 0:
        return f"{days}d {hours}h"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


# ==================== PIP CALCULATIONS ====================

def get_pip_value(symbol: str) -> float:
    """
    Get pip value for a symbol.
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        
    Returns:
        float: Pip value (0.0001 for EURUSD, 0.01 for USDJPY, etc.)
    """
    symbol = normalize_symbol(symbol)
    return config.PIP_VALUES.get(symbol, 0.0001)


def calculate_pips(symbol: str, price1: float, price2: float) -> float:
    """
    Calculate pip distance between two prices.
    
    Args:
        symbol: Trading symbol
        price1: First price
        price2: Second price
        
    Returns:
        float: Pip distance (absolute value)
    """
    pip_value = get_pip_value(symbol)
    pips = abs(price2 - price1) / pip_value
    return round(pips, 1)


def calculate_price_from_pips(symbol: str, base_price: float, pips: float, direction: str = 'up') -> float:
    """
    Calculate price that is X pips away from base price.
    
    Args:
        symbol: Trading symbol
        base_price: Starting price
        pips: Number of pips away
        direction: 'up' or 'down'
        
    Returns:
        float: Calculated price
    """
    pip_value = get_pip_value(symbol)
    
    if direction == 'up':
        new_price = base_price + (pips * pip_value)
    else:
        new_price = base_price - (pips * pip_value)
    
    # Round to appropriate decimal places
    if 'JPY' in symbol:
        return round(new_price, 3)
    elif symbol in ['XAUUSD', 'XAGUSD']:
        return round(new_price, 2)
    else:
        return round(new_price, 5)


def calculate_lot_size(
    account_balance: float,
    risk_percent: float,
    stop_loss_pips: float,
    symbol: str,
    account_currency: str = 'USD'
) -> float:
    """
    Calculate optimal lot size based on risk parameters.
    
    Args:
        account_balance: Account balance in account currency
        risk_percent: Risk percentage (e.g., 1.0 for 1%)
        stop_loss_pips: Stop loss distance in pips
        symbol: Trading symbol
        account_currency: Account currency
        
    Returns:
        float: Calculated lot size
    """
    risk_amount = account_balance * (risk_percent / 100)
    pip_value = get_pip_value(symbol)
    
    # Get contract size
    if symbol in ['XAUUSD', 'XAGUSD']:
        contract_size = config.CONTRACT_SIZES.get(symbol, 100)
    elif 'BTC' in symbol or 'ETH' in symbol:
        contract_size = 1
    else:
        contract_size = config.CONTRACT_SIZES['FOREX']
    
    # Calculate value per pip for 1 lot
    value_per_pip = pip_value * contract_size
    
    # Calculate lot size
    lot_size = risk_amount / (stop_loss_pips * value_per_pip)
    
    # Round to 2 decimal places
    lot_size = round(lot_size, 2)
    
    # Minimum lot size
    if lot_size < 0.01:
        lot_size = 0.01
    
    return lot_size


def calculate_profit_usd(
    symbol: str,
    lot_size: float,
    entry_price: float,
    exit_price: float,
    direction: str
) -> float:
    """
    Calculate profit in USD for a trade.
    
    Args:
        symbol: Trading symbol
        lot_size: Position size in lots
        entry_price: Entry price
        exit_price: Exit price
        direction: 'BUY' or 'SELL'
        
    Returns:
        float: Profit in USD (negative if loss)
    """
    pip_value = get_pip_value(symbol)
    pips = calculate_pips(symbol, entry_price, exit_price)
    
    # Get contract size
    if symbol in ['XAUUSD', 'XAGUSD']:
        contract_size = config.CONTRACT_SIZES.get(symbol, 100)
    elif 'BTC' in symbol or 'ETH' in symbol:
        contract_size = 1
    else:
        contract_size = config.CONTRACT_SIZES['FOREX']
    
    # Calculate profit
    profit = pips * pip_value * contract_size * lot_size
    
    # Invert if SELL trade and price went up
    if direction == 'SELL':
        if exit_price > entry_price:
            profit = -profit
    else:  # BUY
        if exit_price < entry_price:
            profit = -profit
    
    return round(profit, 2)


def calculate_risk_reward(
    entry_price: float,
    stop_loss: float,
    take_profit: float
) -> float:
    """
    Calculate risk-reward ratio.
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        
    Returns:
        float: R:R ratio (e.g., 2.0 for 1:2)
    """
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    
    if risk == 0:
        return 0.0
    
    rr_ratio = reward / risk
    return round(rr_ratio, 2)


# ==================== SYMBOL UTILITIES ====================

def normalize_symbol(broker_symbol: str) -> str:
    """
    Normalize broker-specific symbol to standard format.
    
    Args:
        broker_symbol: Broker's symbol (e.g., 'EURUSD.pro', 'GBPUSDm')
        
    Returns:
        str: Normalized symbol (e.g., 'EURUSD', 'GBPUSD')
    """
    # Remove common suffixes
    for suffix in config.SYMBOL_SUFFIXES:
        if broker_symbol.endswith(suffix):
            broker_symbol = broker_symbol[:-len(suffix)]
    
    # Remove any remaining special characters
    broker_symbol = re.sub(r'[^A-Z0-9]', '', broker_symbol.upper())
    
    return broker_symbol


def denormalize_symbol(standard_symbol: str, symbol_mapping: Optional[Dict[str, str]] = None) -> str:
    """
    Convert standard symbol to broker-specific format.
    
    Args:
        standard_symbol: Standard symbol (e.g., 'EURUSD')
        symbol_mapping: User's symbol mappings from database
        
    Returns:
        str: Broker symbol or standard if no mapping exists
    """
    if symbol_mapping and standard_symbol in symbol_mapping:
        return symbol_mapping[standard_symbol]
    
    return standard_symbol


# ==================== FORMATTING UTILITIES ====================

def format_price(symbol: str, price: float) -> str:
    """
    Format price with appropriate decimal places.
    
    Args:
        symbol: Trading symbol
        price: Price value
        
    Returns:
        str: Formatted price
    """
    if 'JPY' in symbol:
        return f"{price:.3f}"
    elif symbol in ['XAUUSD', 'XAGUSD']:
        return f"{price:.2f}"
    elif 'BTC' in symbol:
        return f"{price:.0f}"
    elif 'ETH' in symbol:
        return f"{price:.2f}"
    else:
        return f"{price:.5f}"


def format_currency(amount: float, currency: str = 'USD') -> str:
    """
    Format currency amount.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        str: Formatted string like "$132.50 USD"
    """
    if amount >= 0:
        return f"${amount:.2f} {currency}"
    else:
        return f"-${abs(amount):.2f} {currency}"


def format_percentage(value: float) -> str:
    """
    Format percentage value.
    
    Args:
        value: Percentage (e.g., 68.5)
        
    Returns:
        str: Formatted string like "68.5%"
    """
    return f"{value:.1f}%"


def format_risk_reward(rr: float) -> str:
    """
    Format risk-reward ratio.
    
    Args:
        rr: R:R ratio (e.g., 2.33)
        
    Returns:
        str: Formatted string like "1:2.33"
    """
    return f"1:{rr:.2f}"


# ==================== TEXT UTILITIES ====================

def replace_forbidden_words(text: str) -> str:
    """
    Replace forbidden trading terms with compliant language.
    
    Args:
        text: Original text
        
    Returns:
        str: Text with replacements
    """
    for forbidden, replacement in config.FORBIDDEN_WORDS.items():
        # Case-insensitive replacement
        pattern = re.compile(re.escape(forbidden), re.IGNORECASE)
        text = pattern.sub(replacement, text)
    
    return text


def add_footer(text: str) -> str:
    """
    Add perpetual compliance footer to message.
    
    Args:
        text: Original message text
        
    Returns:
        str: Message with footer
    """
    return f"{text}\n\n{config.LEGAL_DISCLAIMER}"


def truncate_message(text: str, max_length: int = config.MAX_MESSAGE_LENGTH) -> List[str]:
    """
    Split long message into multiple parts if needed.
    
    Args:
        text: Message text
        max_length: Maximum length per message
        
    Returns:
        list: List of message parts
    """
    if len(text) <= max_length:
        return [text]
    
    parts = []
    current_part = ""
    
    for line in text.split('\n'):
        if len(current_part) + len(line) + 1 <= max_length:
            current_part += line + '\n'
        else:
            if current_part:
                parts.append(current_part.strip())
            current_part = line + '\n'
    
    if current_part:
        parts.append(current_part.strip())
    
    return parts


# ==================== VALIDATION UTILITIES ====================

def validate_risk_percent(risk: float) -> Tuple[bool, str]:
    """
    Validate risk percentage input.
    
    Args:
        risk: Risk percentage value
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if risk < config.MIN_RISK_PERCENT:
        return False, f"Risk must be at least {config.MIN_RISK_PERCENT}%"
    
    if risk > config.MAX_RISK_PERCENT:
        return False, f"Risk cannot exceed {config.MAX_RISK_PERCENT}%"
    
    return True, ""


def validate_symbol(symbol: str) -> bool:
    """
    Check if symbol is in monitored list.
    
    Args:
        symbol: Trading symbol
        
    Returns:
        bool: True if valid
    """
    normalized = normalize_symbol(symbol)
    return normalized in config.MONITORED_SYMBOLS


# ==================== ORDER TYPE DETECTION ====================

def determine_order_type(current_price: float, entry_price: float, symbol: str) -> Dict[str, any]:
    """
    Determine order type based on price distance.
    
    Args:
        current_price: Current market price
        entry_price: Desired entry price
        symbol: Trading symbol
        
    Returns:
        dict: Order type information with description
    """
    distance_pips = calculate_pips(symbol, current_price, entry_price)
    
    if distance_pips <= 2:
        return {
            'type': 'MARKET',
            'description': f'MARKET ORDER - Entry at {format_price(symbol, entry_price)} within {distance_pips:.1f} pips of current price. Trade executes immediately.',
            'expiry_hours': 0
        }
    
    elif distance_pips <= 20:
        direction = 'pullback' if entry_price < current_price else 'rally'
        return {
            'type': 'LIMIT',
            'description': f'LIMIT ORDER - Entry at {format_price(symbol, entry_price)} requires {direction} from current {format_price(symbol, current_price)} ({distance_pips:.1f} pips away). Order will expire in 1 hour if not filled.',
            'expiry_hours': 1
        }
    
    else:
        direction = 'upward' if entry_price > current_price else 'downward'
        return {
            'type': 'STOP',
            'description': f'STOP ORDER - Entry at {format_price(symbol, entry_price)} requires {direction} breakout from current {format_price(symbol, current_price)} ({distance_pips:.1f} pips away). Order will expire in 1 hour if not filled.',
            'expiry_hours': 1
        }