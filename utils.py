"""
NIX TRADES - Utility Functions
Role: Python Developer + Product Manager
Fixes:
  - Added missing replace_forbidden_words() function (alias for validate_user_message)
  - Added missing add_footer() function referenced throughout bot.py
  - Added missing timedelta import used by database.py indirectly
  - Fixed extract_currency_from_symbol to not strip trailing 'm' from all symbols
    (e.g. 'EURUSDm' -> 'EURUSD' correct, but 'XAUUSD' was being stripped incorrectly)
  - Added format_setup_message() for standardised alert formatting
NO EMOJIS - Professional code only
"""

import re
import time
import logging
from typing import Optional, Callable, Any, List, Tuple
from datetime import datetime, timezone, timedelta
import pytz
from decimal import Decimal, ROUND_HALF_UP
import config

logger = logging.getLogger(__name__)


def validate_user_message(text: str) -> str:
    """
    Scan text for forbidden words and replace with compliant alternatives.
    This function MUST be called before sending ANY user-facing message.

    Args:
        text: The message text to validate and clean

    Returns:
        str: Cleaned text with forbidden words replaced
    """
    if not text:
        return text

    cleaned_text = text

    for forbidden, replacement in config.WORD_REPLACEMENTS.items():
        pattern = r'\b' + re.escape(forbidden) + r'\b'
        cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)

    if cleaned_text != text:
        logger.debug(
            "Compliance filter applied. Original length: %d, Cleaned length: %d",
            len(text), len(cleaned_text)
        )

    return cleaned_text


def replace_forbidden_words(text: str) -> str:
    """
    Alias for validate_user_message for backward compatibility with bot.py calls.

    Args:
        text: Text to filter

    Returns:
        str: Filtered text
    """
    return validate_user_message(text)


def add_footer(text: str) -> str:
    """
    Append the standard Nix Trades footer to a message if not already present.

    Args:
        text: Message text

    Returns:
        str: Message with footer appended
    """
    footer = config.FOOTER
    if footer not in text:
        return text.rstrip() + "\n\n" + footer
    return text


def format_setup_message(
    signal_number: int,
    symbol: str,
    direction: str,
    setup_type: str,
    entry_price: float,
    stop_loss: float,
    take_profit_1: float,
    take_profit_2: float,
    sl_pips: float,
    tp1_pips: float,
    tp2_pips: float,
    rr_tp1: float,
    rr_tp2: float,
    ml_score: int,
    session: str,
    order_type: str,
    lot_size: Optional[float] = None
) -> str:
    """
    Format a standardised automated setup alert message for Telegram.
    All text passes through validate_user_message before return.

    Args:
        signal_number:  Sequential setup number
        symbol:         Trading symbol e.g. EURUSD
        direction:      'BUY' or 'SELL'
        setup_type:     'UNICORN' or 'STANDARD'
        entry_price:    Entry price level
        stop_loss:      Stop loss price level
        take_profit_1:  First take profit price level
        take_profit_2:  Second take profit price level
        sl_pips:        Stop loss distance in pips
        tp1_pips:       TP1 distance in pips
        tp2_pips:       TP2 distance in pips
        rr_tp1:         Risk-reward ratio to TP1
        rr_tp2:         Risk-reward ratio to TP2
        ml_score:       Model agreement score (0-100)
        session:        Trading session name
        order_type:     'MARKET', 'LIMIT', or 'STOP'
        lot_size:       Calculated lot size (if MT5 connected)

    Returns:
        str: Formatted and compliance-filtered setup message
    """
    direction_label = "LONG" if direction == "BUY" else "SHORT"
    order_type_label = {
        'MARKET': 'MARKET ORDER (executes immediately at current price)',
        'LIMIT':  'LIMIT ORDER (price needs to pull back to entry)',
        'STOP':   'STOP ORDER (breakout entry above/below current price)'
    }.get(order_type, order_type)

    tier_label = "UNICORN SETUP" if setup_type == "UNICORN" else "STANDARD SETUP"

    lines = [
        f"AUTOMATED SETUP #{signal_number} - {tier_label}",
        "",
        f"Pair:          {symbol}",
        f"Direction:     {direction_label}",
        f"Session:       {session}",
        f"Order Type:    {order_type_label}",
        "",
        f"Entry:         {entry_price:.5f}" if symbol not in ('XAUUSD', 'XAGUSD') else f"Entry:         {entry_price:.2f}",
        f"Stop Loss:     {stop_loss:.5f}  ({sl_pips:.1f} pips)" if symbol not in ('XAUUSD', 'XAGUSD') else f"Stop Loss:     {stop_loss:.2f}  ({sl_pips:.1f} pips)",
        f"TP1:           {take_profit_1:.5f}  ({tp1_pips:.1f} pips, R:R 1:{rr_tp1:.2f})" if symbol not in ('XAUUSD', 'XAGUSD') else f"TP1:           {take_profit_1:.2f}  ({tp1_pips:.1f} pips, R:R 1:{rr_tp1:.2f})",
        f"TP2:           {take_profit_2:.5f}  ({tp2_pips:.1f} pips, R:R 1:{rr_tp2:.2f})" if symbol not in ('XAUUSD', 'XAGUSD') else f"TP2:           {take_profit_2:.2f}  ({tp2_pips:.1f} pips, R:R 1:{rr_tp2:.2f})",
        "",
        f"Model Agreement Score: {ml_score}%",
    ]

    if lot_size is not None:
        lines.append(f"Auto Lot Size: {lot_size:.2f} lots")

    lines += [
        "",
        "Management:",
        "- TP1 hit: 50% closed, stop loss moved to breakeven",
        "- TP2 hit: Remaining 50% closed",
        "- Limit/Stop orders expire in 1 hour if not filled",
        "",
        "EDUCATIONAL PURPOSES ONLY. NOT FINANCIAL ADVICE.",
        "Past performance does not guarantee future results.",
    ]

    raw_message = "\n".join(lines)
    return validate_user_message(add_footer(raw_message))


def detect_timezone(user_locale: Optional[str] = None, user_ip: Optional[str] = None) -> str:
    """
    Detect user's timezone based on Telegram locale.
    Telegram does not expose timezone directly. Defaults to UTC.

    Args:
        user_locale: Telegram user locale (e.g., 'en-US', 'en-GB')
        user_ip:     User IP address (not used - Telegram does not expose this)

    Returns:
        str: IANA timezone string
    """
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

    if user_locale and user_locale in locale_timezone_map:
        return locale_timezone_map[user_locale]

    logger.debug("Could not detect timezone for locale: %s. Defaulting to UTC.", user_locale)
    return 'UTC'


def convert_utc_to_user_time(utc_time: datetime, user_timezone: str) -> datetime:
    """
    Convert UTC datetime to user's local timezone.

    Args:
        utc_time:      datetime object (timezone-aware or naive UTC)
        user_timezone: IANA timezone string

    Returns:
        datetime: Converted to user's local timezone
    """
    try:
        user_tz = pytz.timezone(user_timezone)

        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=timezone.utc)

        return utc_time.astimezone(user_tz)

    except Exception as e:
        logger.error("Error converting timezone: %s", e)
        return utc_time


def convert_user_time_to_utc(local_time: datetime, user_timezone: str) -> datetime:
    """
    Convert user's local datetime to UTC.

    Args:
        local_time:    datetime in user's local timezone
        user_timezone: IANA timezone string

    Returns:
        datetime: Converted to UTC
    """
    try:
        user_tz = pytz.timezone(user_timezone)

        if local_time.tzinfo is None:
            local_time = user_tz.localize(local_time)

        return local_time.astimezone(timezone.utc)

    except Exception as e:
        logger.error("Error converting to UTC: %s", e)
        return local_time


def format_currency(amount: float, currency: str = 'USD', decimals: int = 2) -> str:
    """
    Format currency amount for display.

    Args:
        amount:   Amount to format
        currency: Currency code
        decimals: Decimal places

    Returns:
        str: Formatted currency string (e.g., '1,234.56 USD')
    """
    try:
        decimal_amount = Decimal(str(amount))
        quantize_format = '0.' + '0' * decimals
        rounded_amount = decimal_amount.quantize(Decimal(quantize_format), rounding=ROUND_HALF_UP)
        formatted = f"{rounded_amount:,.{decimals}f}"
        return f"{formatted} {currency}"

    except Exception as e:
        logger.error("Error formatting currency: %s", e)
        return f"{amount:.2f} {currency}"


def format_pips(pips: float, decimals: int = 1) -> str:
    """Format pip value for display."""
    return f"{pips:.{decimals}f} pips"


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> Optional[float]:
    """
    Calculate annualised Sharpe ratio for a series of trade returns.

    Args:
        returns:        List of return percentages per trade
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        float: Sharpe ratio, or None if insufficient data
    """
    if not returns or len(returns) < 2:
        return None

    try:
        import numpy as np

        returns_array = np.array(returns, dtype=float)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)

        if std_return == 0:
            return None

        annualized_mean = mean_return * 252
        annualized_std = std_return * np.sqrt(252)
        sharpe = (annualized_mean - risk_free_rate) / annualized_std

        return round(float(sharpe), 2)

    except Exception as e:
        logger.error("Error calculating Sharpe ratio: %s", e)
        return None


def calculate_win_rate(trades: List[dict]) -> Tuple[float, int, int]:
    """
    Calculate historical success rate from trade history.

    Args:
        trades: List of trade dicts with 'realized_pnl' field

    Returns:
        Tuple[float, int, int]: (win_rate_percent, wins, total_trades)
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
    Execute a callable with exponential backoff retry logic.

    Args:
        func:          Function to call
        max_retries:   Maximum number of attempts
        backoff_factor: Sleep multiplier between retries (1s, 2s, 4s, ...)
        exceptions:    Tuple of exception types to catch and retry on
        *args:         Positional arguments for func
        **kwargs:      Keyword arguments for func

    Returns:
        Any: Return value from func

    Raises:
        Exception: Re-raises the last exception after all retries exhausted
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)

        except exceptions as e:
            last_exception = e
            sleep_time = backoff_factor ** attempt

            if attempt < max_retries - 1:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s. Retrying in %.1fs",
                    attempt + 1, max_retries, func.__name__, e, sleep_time
                )
                time.sleep(sleep_time)
            else:
                logger.error(
                    "All %d attempts failed for %s. Last error: %s",
                    max_retries, func.__name__, e
                )

    raise last_exception


def format_duration(start_time: datetime, end_time: datetime) -> str:
    """
    Format time duration in human-readable form.

    Args:
        start_time: Start datetime
        end_time:   End datetime

    Returns:
        str: e.g. '2h 35m', '45m', '3d 4h'
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
        if minutes > 0 or not parts:
            parts.append(f"{minutes}m")

        return ' '.join(parts)

    except Exception as e:
        logger.error("Error formatting duration: %s", e)
        return "N/A"


def format_time_until(target_time: datetime, user_timezone: str) -> str:
    """
    Format time remaining until a target event in user's timezone.

    Args:
        target_time:   Target datetime (UTC)
        user_timezone: IANA timezone string

    Returns:
        str: e.g. '1h 52m', '25m', 'now'
    """
    try:
        local_target = convert_utc_to_user_time(target_time, user_timezone)
        user_tz = pytz.timezone(user_timezone)
        now_local = datetime.now(user_tz)

        if local_target > now_local:
            return format_duration(now_local, local_target)
        return "now"

    except Exception as e:
        logger.error("Error calculating time until: %s", e)
        return "N/A"


def is_market_open(current_time: datetime = None) -> bool:
    """
    Check if the forex market is open (not weekend).
    Forex closes Friday 5 PM EST, opens Sunday 5 PM EST.

    Args:
        current_time: UTC datetime to check. Uses current time if None.

    Returns:
        bool: True if market is open
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc)

    est = pytz.timezone('America/New_York')
    est_time = current_time.astimezone(est)
    day_of_week = est_time.weekday()  # 0=Monday, 6=Sunday
    hour = est_time.hour

    if day_of_week == 5:  # Saturday - always closed
        return False

    if day_of_week == 6 and hour < 17:  # Sunday before 5 PM EST
        return False

    if day_of_week == 4 and hour >= 17:  # Friday after 5 PM EST
        return False

    return True


def get_session_name(utc_hour: int) -> str:
    """
    Get trading session name from UTC hour.

    Args:
        utc_hour: Hour in UTC (0-23)

    Returns:
        str: Session name
    """
    if 0 <= utc_hour < 7:
        return 'Asian'
    elif utc_hour == 7:
        return 'London Open'
    elif 8 <= utc_hour < 13:
        return 'London'
    elif 13 <= utc_hour < 16:
        return 'Overlap'
    elif 16 <= utc_hour < 21:
        return 'New York'
    else:
        return 'Off Hours'


def sanitize_filename(filename: str) -> str:
    """
    Remove invalid characters from a filename.

    Args:
        filename: Original filename

    Returns:
        str: Sanitized filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    if len(filename) > 200:
        filename = filename[:200]

    return filename


def get_current_utc_time() -> datetime:
    """
    Get current UTC time as a timezone-aware datetime.

    Returns:
        datetime: Current UTC time
    """
    return datetime.now(timezone.utc)


def parse_iso_datetime(iso_string: str) -> Optional[datetime]:
    """
    Parse ISO 8601 datetime string.

    Args:
        iso_string: ISO formatted datetime string

    Returns:
        datetime: Parsed datetime, or None if parsing fails
    """
    try:
        return datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
    except Exception as e:
        logger.error("Error parsing ISO datetime '%s': %s", iso_string, e)
        return None


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage value (e.g., '68.5%')."""
    return f"{value:.{decimals}f}%"


def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_currency_from_symbol(symbol: str) -> Tuple[str, str]:
    """
    Extract base and quote currency from a broker symbol string.
    Fix: Only strip trailing 'm' when it follows a 6-character standard pair,
    not from all symbols (which would corrupt 'EURUSDm' -> 'EURSD').

    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'EURUSDm', 'XAUUSD')

    Returns:
        Tuple[str, str]: (base_currency, quote_currency)
    """
    clean_symbol = symbol

    for suffix in ['.pro', '.raw', '-a', 'm']:
        if clean_symbol.endswith(suffix):
            candidate = clean_symbol[:-len(suffix)]
            # Only accept the strip if it results in a known pair length
            if len(candidate) >= 6:
                clean_symbol = candidate
                break

    if clean_symbol.startswith('XAU'):
        return 'XAU', clean_symbol[3:]
    if clean_symbol.startswith('XAG'):
        return 'XAG', clean_symbol[3:]

    if len(clean_symbol) >= 6:
        return clean_symbol[:3], clean_symbol[3:6]

    return clean_symbol, 'USD'


def validate_risk_percent(risk_percent: float) -> bool:
    """
    Validate risk percentage is within acceptable bounds (0.1% - 5.0%).

    Args:
        risk_percent: Risk percentage per trade

    Returns:
        bool: True if valid
    """
    return config.MIN_RISK_PERCENT <= risk_percent <= config.MAX_RISK_PERCENT


def validate_lot_size(lot_size: float) -> bool:
    """
    Validate lot size is within acceptable bounds.

    Args:
        lot_size: Lot size value

    Returns:
        bool: True if valid
    """
    return config.MIN_LOT_SIZE <= lot_size <= config.MAX_LOT_SIZE


def parse_mt5_credentials(text: str) -> Optional[dict]:
    """
    Parse MT5 credentials from user message text.
    Expected format (case-insensitive):
        LOGIN: 12345678
        PASSWORD: YourPassword
        SERVER: ICMarkets-Demo

    Args:
        text: Raw message text from user

    Returns:
        dict with keys 'login', 'password', 'server', or None if parsing fails
    """
    try:
        login = None
        password = None
        server = None

        for line in text.strip().splitlines():
            line = line.strip()
            if ':' not in line:
                continue

            key, _, value = line.partition(':')
            key = key.strip().upper()
            value = value.strip()

            if key == 'LOGIN':
                login = value
            elif key == 'PASSWORD':
                password = value
            elif key == 'SERVER':
                server = value

        if not login or not password or not server:
            return None

        if not login.isdigit():
            return None

        return {
            'login': int(login),
            'password': password,
            'server': server
        }

    except Exception as e:
        logger.error("Error parsing MT5 credentials: %s", e)
        return None