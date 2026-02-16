"""
Utility functions for Nix Trades Telegram Bot
Includes forbidden word filtering, timezone detection, and helper functions
NO EMOJIS - Professional code only
"""

import re
import time
import logging
from typing import Optional, Callable, Any, List, Tuple
from datetime import datetime, timezone
import pytz
from decimal import Decimal, ROUND_HALF_UP
import config

logger = logging.getLogger(__name__)


def validate_user_message(text: str) -> str:
    """
    CRITICAL FUNCTION: Scans text for forbidden words and replaces them with compliant alternatives.
    This function MUST be called before sending ANY user-facing message.
    
    Args:
        text: The message text to validate and clean
        
    Returns:
        str: Cleaned text with forbidden words replaced
        
    Example:
        >>> validate_user_message("This signal has a 90% win rate!")
        "This automated setup has a historical success rate of 90% (past performance does not guarantee future results)!"
    """
    if not text:
        return text
    
    cleaned_text = text
    
    # Replace each forbidden word/phrase with its approved alternative
    # Use word boundaries to avoid partial replacements
    for forbidden, replacement in config.WORD_REPLACEMENTS.items():
        # Case-insensitive replacement with word boundaries
        pattern = r'\b' + re.escape(forbidden) + r'\b'
        cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
    
    # Log if replacements were made (for monitoring)
    if cleaned_text != text:
        logger.info(f"Forbidden words filtered in message. Original length: {len(text)}, Cleaned length: {len(cleaned_text)}")
    
    return cleaned_text


def detect_timezone(user_locale: Optional[str] = None, user_ip: Optional[str] = None) -> str:
    """
    Detect user's timezone based on locale or IP geolocation.
    
    Args:
        user_locale: Telegram user locale (e.g., 'en-US', 'en-GB')
        user_ip: User's IP address for geolocation fallback
        
    Returns:
        str: IANA timezone string (e.g., 'America/New_York', 'Europe/London')
        
    Note:
        Telegram API doesn't provide timezone directly. This function makes best guess.
        Default to UTC if detection fails. User can manually set in /settings.
    """
    # Locale to timezone mapping (common cases)
    locale_timezone_map = {
        'en-US': 'America/New_York',
        'en-GB': 'Europe/London',
        'en-CA': 'America/Toronto',
        'en-AU': 'Australia/Sydney',
        'en-NZ': 'Pacific/Auckland',
        'en-ZA': 'Africa/Johannesburg',
        'en-NG': 'Africa/Lagos',
        'fr-FR': 'Europe/Paris',
        'de-DE': 'Europe/Berlin',
        'es-ES': 'Europe/Madrid',
        'it-IT': 'Europe/Rome',
        'pt-BR': 'America/Sao_Paulo',
        'ja-JP': 'Asia/Tokyo',
        'zh-CN': 'Asia/Shanghai',
        'ko-KR': 'Asia/Seoul',
        'ar-SA': 'Asia/Riyadh',
        'hi-IN': 'Asia/Kolkata',
        'ru-RU': 'Europe/Moscow'
    }
    
    # Try locale-based detection first
    if user_locale and user_locale in locale_timezone_map:
        return locale_timezone_map[user_locale]
    
    # Fallback: Default to UTC (user will be prompted to set manually)
    logger.warning(f"Could not detect timezone for locale: {user_locale}. Defaulting to UTC.")
    return 'UTC'


def convert_utc_to_user_time(utc_time: datetime, user_timezone: str) -> datetime:
    """
    Convert UTC datetime to user's local timezone.
    
    Args:
        utc_time: datetime object in UTC (timezone-aware)
        user_timezone: IANA timezone string (e.g., 'America/New_York')
        
    Returns:
        datetime: Converted to user's local timezone
        
    Example:
        >>> utc = datetime(2024, 2, 8, 14, 30, tzinfo=timezone.utc)
        >>> convert_utc_to_user_time(utc, 'America/New_York')
        datetime(2024, 2, 8, 9, 30, tzinfo=<DstTzInfo 'America/New_York' EST-1 day, 19:00:00 STD>)
    """
    try:
        user_tz = pytz.timezone(user_timezone)
        
        # If utc_time is naive, assume it's UTC
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=timezone.utc)
        
        # Convert to user timezone
        local_time = utc_time.astimezone(user_tz)
        return local_time
    
    except Exception as e:
        logger.error(f"Error converting timezone: {e}")
        return utc_time  # Return original if conversion fails


def convert_user_time_to_utc(local_time: datetime, user_timezone: str) -> datetime:
    """
    Convert user's local datetime to UTC.
    
    Args:
        local_time: datetime object in user's local timezone
        user_timezone: IANA timezone string
        
    Returns:
        datetime: Converted to UTC
    """
    try:
        user_tz = pytz.timezone(user_timezone)
        
        # Localize if naive
        if local_time.tzinfo is None:
            local_time = user_tz.localize(local_time)
        
        # Convert to UTC
        utc_time = local_time.astimezone(timezone.utc)
        return utc_time
    
    except Exception as e:
        logger.error(f"Error converting to UTC: {e}")
        return local_time


def format_currency(amount: float, currency: str = 'USD', decimals: int = 2) -> str:
    """
    Format currency amount for display.
    
    Args:
        amount: Amount to format
        currency: Currency code (default: 'USD')
        decimals: Number of decimal places
        
    Returns:
        str: Formatted currency string
        
    Example:
        >>> format_currency(1234.56)
        '1,234.56 USD'
    """
    try:
        # Use Decimal for precision
        decimal_amount = Decimal(str(amount))
        
        # Round to specified decimals
        quantize_format = '0.' + '0' * decimals
        rounded_amount = decimal_amount.quantize(Decimal(quantize_format), rounding=ROUND_HALF_UP)
        
        # Format with thousands separator
        formatted = f"{rounded_amount:,.{decimals}f}"
        
        return f"{formatted} {currency}"
    
    except Exception as e:
        logger.error(f"Error formatting currency: {e}")
        return f"{amount:.2f} {currency}"


def format_pips(pips: float, decimals: int = 1) -> str:
    """
    Format pip value for display.
    
    Args:
        pips: Pip value
        decimals: Decimal places
        
    Returns:
        str: Formatted pip string
        
    Example:
        >>> format_pips(25.5)
        '25.5 pips'
    """
    return f"{pips:.{decimals}f} pips"


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> Optional[float]:
    """
    Calculate Sharpe ratio for a series of trade returns.
    
    Args:
        returns: List of return percentages (e.g., [2.5, -1.2, 3.1, ...])
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        float: Sharpe ratio, or None if insufficient data
        
    Formula:
        Sharpe = (Mean Return - Risk-Free Rate) / Standard Deviation
    """
    if not returns or len(returns) < 2:
        return None
    
    try:
        import numpy as np
        
        # Convert to numpy array
        returns_array = np.array(returns)
        
        # Calculate mean and std
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)  # Sample std dev
        
        if std_return == 0:
            return None
        
        # Annualize (assuming daily returns)
        annualized_mean = mean_return * 252  # Trading days per year
        annualized_std = std_return * np.sqrt(252)
        
        # Calculate Sharpe ratio
        sharpe = (annualized_mean - risk_free_rate) / annualized_std
        
        return round(sharpe, 2)
    
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {e}")
        return None


def calculate_win_rate(trades: List[dict]) -> Tuple[float, int, int]:
    """
    Calculate historical success rate from trade history.
    
    Args:
        trades: List of trade dictionaries with 'realized_pnl' field
        
    Returns:
        Tuple[float, int, int]: (win_rate_percent, wins, total_trades)
        
    Note:
        Always append "past performance does not guarantee future results" when displaying
    """
    if not trades:
        return 0.0, 0, 0
    
    wins = sum(1 for trade in trades if trade.get('realized_pnl', 0) > 0)
    total = len(trades)
    win_rate = (wins / total) * 100 if total > 0 else 0.0
    
    return round(win_rate, 1), wins, total


def try_with_retry(
    func: Callable,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: Tuple = (Exception,),
    *args,
    **kwargs
) -> Any:
    """
    Execute function with exponential backoff retry logic.
    
    Args:
        func: Function to execute
        max_retries: Maximum retry attempts
        backoff_factor: Multiplier for delay (1s, 2s, 4s, ...)
        exceptions: Tuple of exceptions to catch and retry
        *args, **kwargs: Arguments to pass to func
        
    Returns:
        Function result if successful
        
    Raises:
        Last exception if all retries fail
        
    Example:
        >>> result = try_with_retry(mt5.order_send, max_retries=3, request=order_request)
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        
        except exceptions as e:
            last_exception = e
            
            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff
                delay = (backoff_factor ** attempt)
                
                # Add jitter (0-500ms) to prevent thundering herd
                import random
                jitter = random.uniform(0, 0.5)
                total_delay = delay + jitter
                
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {total_delay:.2f}s..."
                )
                
                time.sleep(total_delay)
            else:
                logger.error(
                    f"All {max_retries} attempts failed for {func.__name__}. "
                    f"Last error: {e}"
                )
    
    # All retries exhausted
    raise last_exception


def format_duration(start_time: datetime, end_time: datetime) -> str:
    """
    Format time duration in human-readable format.
    
    Args:
        start_time: Start datetime
        end_time: End datetime
        
    Returns:
        str: Formatted duration (e.g., '2h 35m', '45m', '3d 4h')
        
    Example:
        >>> start = datetime(2024, 2, 8, 10, 0)
        >>> end = datetime(2024, 2, 8, 12, 35)
        >>> format_duration(start, end)
        '2h 35m'
    """
    try:
        duration = end_time - start_time
        
        days = duration.days
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or not parts:  # Show minutes if nothing else
            parts.append(f"{minutes}m")
        
        return ' '.join(parts)
    
    except Exception as e:
        logger.error(f"Error formatting duration: {e}")
        return "N/A"


def format_time_until(target_time: datetime, user_timezone: str) -> str:
    """
    Format time remaining until target event in user's timezone.
    
    Args:
        target_time: Target datetime (UTC)
        user_timezone: User's IANA timezone string
        
    Returns:
        str: Time remaining (e.g., '1h 52m', '25m', '2h 15m')
    """
    try:
        # Convert target to user timezone
        local_target = convert_utc_to_user_time(target_time, user_timezone)
        
        # Get current time in user timezone
        user_tz = pytz.timezone(user_timezone)
        now_local = datetime.now(user_tz)
        
        # Calculate difference
        if local_target > now_local:
            return format_duration(now_local, local_target)
        else:
            return "now"
    
    except Exception as e:
        logger.error(f"Error calculating time until: {e}")
        return "N/A"


def is_market_open(current_time: datetime = None) -> bool:
    """
    Check if forex market is open (not weekend).
    
    Args:
        current_time: datetime to check (UTC). If None, uses current time.
        
    Returns:
        bool: True if market is open, False if weekend
        
    Note:
        Forex market closes Friday 5 PM EST, opens Sunday 5 PM EST
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    
    # Convert to EST for forex market hours
    est = pytz.timezone('America/New_York')
    est_time = current_time.astimezone(est)
    
    day_of_week = est_time.weekday()  # 0=Monday, 6=Sunday
    hour = est_time.hour
    
    # Market closed Saturday (all day) and Sunday before 5 PM
    if day_of_week == 5:  # Saturday
        return False
    
    if day_of_week == 6 and hour < 17:  # Sunday before 5 PM
        return False
    
    # Market closed Friday after 5 PM
    if day_of_week == 4 and hour >= 17:  # Friday after 5 PM
        return False
    
    return True


def get_session_name(utc_hour: int) -> str:
    """
    Get trading session name based on UTC hour.
    
    Args:
        utc_hour: Hour in UTC (0-23)
        
    Returns:
        str: Session name ('Asian', 'London', 'New York', 'Overlap', etc.)
    """
    if 0 <= utc_hour < 7:
        return 'Asian'
    elif utc_hour == 7:
        return 'London Open'
    elif 8 <= utc_hour < 13:
        return 'London'
    elif 13 <= utc_hour < 16:
        return 'Overlap'  # London + New York
    elif 16 <= utc_hour < 21:
        return 'New York'
    else:
        return 'Off Hours'


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename safe for filesystem
    """
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    max_length = 200
    if len(filename) > max_length:
        filename = filename[:max_length]
    
    return filename


def get_current_utc_time() -> datetime:
    """
    Get current UTC time as timezone-aware datetime.
    
    Returns:
        datetime: Current UTC time
    """
    return datetime.now(timezone.utc)


def parse_iso_datetime(iso_string: str) -> Optional[datetime]:
    """
    Parse ISO 8601 datetime string to datetime object.
    
    Args:
        iso_string: ISO formatted datetime string
        
    Returns:
        datetime: Parsed datetime, or None if parsing fails
    """
    try:
        return datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
    except Exception as e:
        logger.error(f"Error parsing ISO datetime '{iso_string}': {e}")
        return None


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format percentage value for display.
    
    Args:
        value: Percentage value (e.g., 68.5)
        decimals: Decimal places
        
    Returns:
        str: Formatted percentage (e.g., '68.5%')
    """
    return f"{value:.{decimals}f}%"


def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_currency_from_symbol(symbol: str) -> Tuple[str, str]:
    """
    Extract base and quote currency from symbol.
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'XAUUSD')
        
    Returns:
        Tuple[str, str]: (base_currency, quote_currency)
        
    Example:
        >>> extract_currency_from_symbol('EURUSD')
        ('EUR', 'USD')
        >>> extract_currency_from_symbol('XAUUSD')
        ('XAU', 'USD')
    """
    # Remove common suffixes
    clean_symbol = symbol.replace('.pro', '').replace('.raw', '').replace('-a', '').replace('m', '')
    
    # Special cases
    if clean_symbol.startswith('XAU'):
        return 'XAU', clean_symbol[3:]
    if clean_symbol.startswith('XAG'):
        return 'XAG', clean_symbol[3:]
    
    # Standard forex pairs (6 characters)
    if len(clean_symbol) >= 6:
        return clean_symbol[:3], clean_symbol[3:6]
    
    # Fallback
    return clean_symbol, 'USD'


def validate_risk_percent(risk_percent: float) -> bool:
    """
    Validate risk percentage is within acceptable range.
    
    Args:
        risk_percent: Risk percentage per trade
        
    Returns:
        bool: True if valid (0.1% - 5.0%)
    """
    return 0.1 <= risk_percent <= 5.0


def validate_lot_size(lot_size: float) -> bool:
    """
    Validate lot size is within acceptable range.
    
    Args:
        lot_size: Lot size value
        
    Returns:
        bool: True if valid (0.01 - 10.0)
    """
    return config.MIN_LOT_SIZE <= lot_size <= config.MAX_LOT_SIZE