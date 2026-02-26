"""
NIX TRADES - Utility Functions
Role: Python Developer + Product Manager

Changes in this version:
  - Added get_session() - returns current trading session name (called by scheduler)
  - Added calculate_price_from_pips() - used by smc_strategy.calculate_take_profits
  - Added determine_order_type() - used by scheduler broadcast formatting
  - Added calculate_time_until() - used by news_fetcher integration
  - Added format_risk_reward() - used by scheduler setup formatting
  - All existing functions preserved and unchanged

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


# ==================== COMPLIANCE / MESSAGE SANITISATION ====================

def validate_user_message(text: str) -> str:
    """
    Scan text for forbidden words and replace with compliant alternatives.
    MUST be called before sending ANY user-facing message.
    """
    if not text:
        return text
    cleaned = text
    for forbidden, replacement in config.WORD_REPLACEMENTS.items():
        pattern = r'\b' + re.escape(forbidden) + r'\b'
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    if cleaned != text:
        logger.debug(
            "Compliance filter applied. Original: %d chars, Cleaned: %d chars",
            len(text), len(cleaned))
    return cleaned


def replace_forbidden_words(text: str) -> str:
    """Alias for validate_user_message for backward compatibility."""
    return validate_user_message(text)


def add_footer(text: str) -> str:
    """Append the standard footer to a message if not already present."""
    footer = config.FOOTER
    if footer not in text:
        return text.rstrip() + "\n\n" + footer
    return text


# ==================== SESSION ====================

def get_session() -> str:
    """
    Return the current trading session name based on UTC time.
    Called by scheduler.py when building setup_data.

    Returns:
        str: One of 'Asian', 'London Open', 'London', 'Overlap', 'New York', 'Off Hours'
    """
    return get_session_name(datetime.now(timezone.utc).hour)


def get_session_name(utc_hour: int) -> str:
    """
    Get trading session name from a UTC hour integer.

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


# ==================== PRICE / PIP HELPERS ====================

def get_pip_value(symbol: str) -> float:
    """
    Return the pip size for a given symbol.

    Args:
        symbol: Standard or broker-specific symbol name

    Returns:
        float: Pip size (e.g. 0.0001 for EURUSD, 0.01 for USDJPY, 0.10 for XAUUSD)
    """
    clean = _clean_symbol(symbol)
    return config.PIP_SIZES.get(clean, 0.0001)


def calculate_pips(symbol: str, price1: float, price2: float) -> float:
    """
    Calculate the pip distance between two prices for a given symbol.

    Args:
        symbol: Trading symbol
        price1: First price level
        price2: Second price level

    Returns:
        float: Absolute pip distance
    """
    pip_val = get_pip_value(symbol)
    if pip_val <= 0:
        return 0.0
    return abs(price1 - price2) / pip_val


def calculate_price_from_pips(symbol: str, base_price: float,
                               pips: float, direction: str) -> float:
    """
    Convert a pip count to an actual price level.

    Args:
        symbol:     Trading symbol (used to determine pip size)
        base_price: Starting price
        pips:       Number of pips to move
        direction:  'up' or 'down'

    Returns:
        float: Resulting price
    """
    pip_val = get_pip_value(symbol)
    delta   = pips * pip_val
    if direction == 'up':
        return base_price + delta
    return base_price - delta



def format_price(symbol: str, price: float) -> str:
    """
    Format a price with the correct number of decimal places for the symbol.

    Args:
        symbol: Trading symbol
        price:  Price to format

    Returns:
        str: Formatted price string
    """
    clean = _clean_symbol(symbol)
    if clean in ('XAUUSD', 'XAGUSD'):
        return f"{price:.2f}"
    if clean in ('USDJPY', 'EURJPY', 'GBPJPY'):
        return f"{price:.3f}"
    return f"{price:.5f}"


def _clean_symbol(symbol: str) -> str:
    """Strip broker suffixes to get the canonical symbol name."""
    for suffix in config.SYMBOL_SUFFIXES:
        if symbol.endswith(suffix) and len(symbol) - len(suffix) >= 6:
            return symbol[:-len(suffix)]
    return symbol


# ==================== ORDER TYPE DETECTION ====================

def determine_order_type(
    current_price: float,
    entry_price: float,
    symbol: str,
) -> dict:
    """
    Determine whether a setup requires a MARKET, LIMIT, or STOP order
    based on the gap between current price and intended entry.

    Args:
        current_price: Live bid or ask price
        entry_price:   Intended entry price from the POI
        symbol:        Trading symbol (used to get pip value)

    Returns:
        dict with keys 'type' (str) and 'description' (str)
    """
    pip_val  = get_pip_value(symbol)
    distance = abs(current_price - entry_price)
    pips     = distance / pip_val if pip_val > 0 else 0.0

    if pips <= config.MARKET_ORDER_THRESHOLD_PIPS:
        return {
            'type':        'MARKET',
            'description': 'MARKET ORDER - executes immediately at current price.',
        }
    elif pips <= config.LIMIT_ORDER_THRESHOLD_PIPS:
        return {
            'type':        'LIMIT',
            'description': 'LIMIT ORDER - price must pull back to entry before filling.',
        }
    else:
        return {
            'type':        'STOP',
            'description': 'STOP ORDER - breakout entry, fills if price reaches entry.',
        }


def format_risk_reward(rr: float) -> str:
    """Format a decimal RR ratio as '1:X.XX' string."""
    return "1:%.2f" % rr


# ==================== TIME HELPERS ====================

def calculate_time_until(target_time: datetime) -> str:
    """
    Calculate a human-readable countdown to a future UTC datetime.
    Used by news_fetcher / scheduler for news blackout warnings.

    Args:
        target_time: Future datetime (timezone-aware UTC preferred)

    Returns:
        str: e.g. '1h 25m', '45m', 'now'
    """
    try:
        now = datetime.now(timezone.utc)
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)
        delta = target_time - now
        if delta.total_seconds() <= 0:
            return 'now'
        total_minutes = int(delta.total_seconds() // 60)
        hours, minutes = divmod(total_minutes, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
    except Exception as e:
        logger.error("Error calculating time until: %s", e)
        return 'unknown'

def get_session(utc_dt=None) -> str:
    """
    Return the current trading session name.
    Wraps get_session_name with the current UTC hour.
    Called by scheduler and smc_strategy without arguments.
    """
    if utc_dt is not None and hasattr(utc_dt, 'hour'):
        return get_session_name(utc_dt.hour)
    from datetime import datetime, timezone
    return get_session_name(datetime.now(timezone.utc).hour)


def get_current_utc_time() -> datetime:
    """Return current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def parse_iso_datetime(iso_string: str) -> Optional[datetime]:
    """Parse ISO 8601 datetime string."""
    try:
        return datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
    except Exception as e:
        logger.error("Error parsing ISO datetime '%s': %s", iso_string, e)
        return None


def detect_timezone(
    user_locale: Optional[str] = None,
    user_ip:     Optional[str] = None,
) -> str:
    """
    Guess user timezone from Telegram locale. Defaults to UTC.

    Args:
        user_locale: Telegram locale string (e.g. 'en-GB')
        user_ip:     Not used (Telegram does not expose user IP)

    Returns:
        str: IANA timezone string
    """
    locale_map = {
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
        'ru-RU': 'Europe/Moscow',
    }
    if user_locale and user_locale in locale_map:
        return locale_map[user_locale]
    return 'UTC'


def convert_utc_to_user_time(utc_time: datetime, user_timezone: str) -> datetime:
    """Convert a UTC datetime to the user's local timezone."""
    try:
        tz = pytz.timezone(user_timezone)
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=timezone.utc)
        return utc_time.astimezone(tz)
    except Exception as e:
        logger.error("Error converting timezone: %s", e)
        return utc_time


def convert_user_time_to_utc(local_time: datetime, user_timezone: str) -> datetime:
    """Convert a local datetime to UTC."""
    try:
        tz = pytz.timezone(user_timezone)
        if local_time.tzinfo is None:
            local_time = tz.localize(local_time)
        return local_time.astimezone(timezone.utc)
    except Exception as e:
        logger.error("Error converting to UTC: %s", e)
        return local_time


def format_time_until(target_time: datetime, user_timezone: str) -> str:
    """Format time remaining until target_time in user's timezone."""
    try:
        tz        = pytz.timezone(user_timezone)
        now_local = datetime.now(tz)
        local_tgt = convert_utc_to_user_time(target_time, user_timezone)
        if local_tgt > now_local:
            return format_duration(now_local, local_tgt)
        return 'now'
    except Exception as e:
        logger.error("Error calculating time until: %s", e)
        return 'N/A'


# ==================== FORMATTING ====================

def format_currency(amount: float, currency: str = 'USD', decimals: int = 2) -> str:
    """Format a currency amount for display."""
    try:
        q = '0.' + '0' * decimals
        rounded = Decimal(str(amount)).quantize(Decimal(q), rounding=ROUND_HALF_UP)
        return f"{rounded:,.{decimals}f} {currency}"
    except Exception as e:
        logger.error("Error formatting currency: %s", e)
        return f"{amount:.2f} {currency}"


def format_pips(pips: float, decimals: int = 1) -> str:
    """Format a pip value for display."""
    return f"{pips:.{decimals}f} pips"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a percentage value (e.g. '68.5%')."""
    return f"{value:.{decimals}f}%"


def format_duration(start_time: datetime, end_time: datetime) -> str:
    """
    Format a time duration in human-readable form.

    Returns:
        str: e.g. '2h 35m', '45m', '3d 4h'
    """
    try:
        delta   = end_time - start_time
        days    = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        parts = []
        if days > 0:    parts.append(f"{days}d")
        if hours > 0:   parts.append(f"{hours}h")
        if minutes > 0 or not parts: parts.append(f"{minutes}m")
        return ' '.join(parts)
    except Exception as e:
        logger.error("Error formatting duration: %s", e)
        return 'N/A'


def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# ==================== SETUP MESSAGE FORMAT ====================

def format_setup_message(
    signal_number:  int,
    symbol:         str,
    direction:      str,
    setup_type:     str,
    entry_price:    float,
    stop_loss:      float,
    take_profit_1:  float,
    take_profit_2:  float,
    sl_pips:        float,
    tp1_pips:       float,
    tp2_pips:       float,
    rr_tp1:         float,
    rr_tp2:         float,
    ml_score:       int,
    session:        str,
    order_type:     str,
    lot_size:       Optional[float] = None,
) -> str:
    """
    Format a standardised automated setup alert message for Telegram.
    All text passes through validate_user_message before return.
    """
    direction_label  = 'LONG' if direction == 'BUY' else 'SHORT'
    order_label      = {
        'MARKET': 'MARKET ORDER - executes immediately',
        'LIMIT':  'LIMIT ORDER - waits for pullback to entry',
        'STOP':   'STOP ORDER - breakout entry',
    }.get(order_type, order_type)
    tier_label = 'UNICORN SETUP' if 'UNICORN' in setup_type.upper() or 'PREMIUM' in setup_type.upper() else 'STANDARD SETUP'

    fp = format_price
    lines = [
        f"AUTOMATED SETUP #{signal_number} - {tier_label}",
        "",
        f"Pair:        {symbol}",
        f"Direction:   {direction_label}",
        f"Session:     {session}",
        f"Order Type:  {order_label}",
        "",
        f"Entry:       {fp(symbol, entry_price)}",
        f"Stop Loss:   {fp(symbol, stop_loss)}  ({sl_pips:.1f} pips)",
        f"TP1:         {fp(symbol, take_profit_1)}  ({tp1_pips:.1f} pips, R:R {format_risk_reward(rr_tp1)})",
        f"TP2:         {fp(symbol, take_profit_2)}  ({tp2_pips:.1f} pips, R:R {format_risk_reward(rr_tp2)})",
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
        "- Orders expire in 1 hour if not filled",
        "",
        "EDUCATIONAL PURPOSES ONLY. NOT FINANCIAL ADVICE.",
        "Past performance does not guarantee future results.",
        "",
        config.FOOTER,
    ]
    return validate_user_message("\n".join(lines))


# ==================== VALIDATION ====================

def validate_risk_percent(risk_percent: float) -> bool:
    """Return True if risk_percent is within acceptable bounds."""
    return config.MIN_RISK_PERCENT <= risk_percent <= config.MAX_RISK_PERCENT


def validate_lot_size(lot_size: float) -> bool:
    """Return True if lot_size is within acceptable bounds."""
    return config.MIN_LOT_SIZE <= lot_size <= config.MAX_LOT_SIZE


def parse_mt5_credentials(text: str) -> Optional[dict]:
    """
    Parse MT5 credentials from user message text.
    Expected format:
        LOGIN: 12345678
        PASSWORD: YourPassword
        SERVER: ICMarkets-Demo

    Returns:
        dict with 'login', 'password', 'server' or None
    """
    try:
        login = password = server = None
        for line in text.strip().splitlines():
            line = line.strip()
            if ':' not in line:
                continue
            key, _, value = line.partition(':')
            key   = key.strip().upper()
            value = value.strip()
            if key == 'LOGIN':    login    = value
            elif key == 'PASSWORD': password = value
            elif key == 'SERVER':  server   = value

        if not login or not password or not server:
            return None
        if not login.isdigit():
            return None
        return {'login': int(login), 'password': password, 'server': server}

    except Exception as e:
        logger.error("Error parsing MT5 credentials: %s", e)
        return None


# ==================== MARKET STATUS ====================

def is_market_open(current_time: Optional[datetime] = None) -> bool:
    """
    Return True if the forex market is currently open (not weekend).
    Forex closes Friday 5 PM EST, opens Sunday 5 PM EST.
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    est = pytz.timezone('America/New_York')
    est_time  = current_time.astimezone(est)
    weekday   = est_time.weekday()  # 0=Monday, 6=Sunday
    hour      = est_time.hour
    if weekday == 5: return False                       # Saturday
    if weekday == 6 and hour < 17: return False         # Sunday before 5 PM
    if weekday == 4 and hour >= 17: return False        # Friday after 5 PM
    return True


# ==================== STATISTICS ====================

def calculate_win_rate(trades: List[dict]) -> Tuple[float, int, int]:
    """
    Calculate historical success rate from a list of closed trades.

    Returns:
        Tuple: (win_rate_percent, wins, total_trades)
    """
    if not trades:
        return 0.0, 0, 0
    wins  = sum(1 for t in trades if (t.get('realized_pnl') or 0) > 0)
    total = len(trades)
    rate  = (wins / total * 100) if total > 0 else 0.0
    return round(rate, 1), wins, total


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.02,
) -> Optional[float]:
    """
    Calculate the annualised Sharpe ratio from a list of trade returns.

    Returns:
        float: Sharpe ratio, or None if not enough data
    """
    if not returns or len(returns) < 2:
        return None
    try:
        import numpy as np
        arr  = np.array(returns, dtype=float)
        mean = np.mean(arr)
        std  = np.std(arr, ddof=1)
        if std == 0:
            return None
        sharpe = (mean * 252 - risk_free_rate) / (std * (252 ** 0.5))
        return round(float(sharpe), 2)
    except Exception as e:
        logger.error("Error calculating Sharpe ratio: %s", e)
        return None


# ==================== LOT SIZE ====================

def calculate_lot_size(
    account_balance: float,
    risk_percent:    float,
    sl_pips:         float,
    symbol:          str,
    account_currency: str = 'USD',
    exchange_rates:  Optional[dict] = None,
) -> float:
    """
    Calculate lot size from account balance, risk percentage, and stop-loss pips.

    Standard formula:
        risk_amount   = balance * risk_percent / 100
        pip_val_per_lot = pip_value_in_account_currency per standard lot
        lots          = risk_amount / (sl_pips * pip_val_per_lot)

    Args:
        account_balance:   Account balance in account_currency
        risk_percent:      Risk per trade as a percentage (e.g. 1.0 = 1%)
        sl_pips:           Stop-loss distance in pips
        symbol:            Trading symbol
        account_currency:  Account base currency ('USD', 'GBP', etc.)
        exchange_rates:    Dict of rates like {'EURUSD': 1.0850} for currency conversion

    Returns:
        float: Lot size rounded to 2 decimal places, clamped to MIN/MAX
    """
    try:
        if account_balance <= 0 or risk_percent <= 0 or sl_pips <= 0:
            return config.MIN_LOT_SIZE

        risk_amount = account_balance * risk_percent / 100.0
        pip_val     = get_pip_value(symbol)

        # Approximate pip value per lot in account currency
        # For USD-quoted pairs (EURUSD, GBPUSD): $10 per pip per standard lot
        # For JPY pairs (USDJPY, EURJPY, GBPJPY): varies by rate
        # Simplified formula that is accurate enough for lot sizing
        if symbol.endswith('JPY') or symbol.endswith('JPY.pro'):
            # 1 lot = 100,000 units; pip = 0.01; pip_value = 1000 JPY / rate
            rate = (exchange_rates or {}).get('USDJPY', 150.0)
            pip_value_per_lot = 1000.0 / rate
        elif symbol.startswith('XAU'):
            # Gold: 1 lot = 100 oz; pip = 0.10; pip_value = $10
            pip_value_per_lot = 10.0
        elif symbol.startswith('XAG'):
            # Silver: 1 lot = 5000 oz; pip = 0.01; pip_value = $50
            pip_value_per_lot = 50.0
        elif symbol[:3] == account_currency:
            # e.g. USDCAD with USD account: pip_value needs rate conversion
            rate = (exchange_rates or {}).get(symbol[:6], 1.0)
            pip_value_per_lot = (pip_val * 100000) / rate
        else:
            # Default: assume USD-quoted pair ($10 per pip per lot)
            pip_value_per_lot = 10.0

        lots = risk_amount / (sl_pips * pip_value_per_lot)
        lots = round(lots, 2)
        lots = max(config.MIN_LOT_SIZE, min(lots, config.MAX_LOT_SIZE))
        return lots

    except Exception as e:
        logger.error("Error calculating lot size: %s", e)
        return config.MIN_LOT_SIZE


# ==================== RETRY UTILITY ====================

def try_with_retry(
    func:            Callable,
    max_retries:     int   = 3,
    backoff_factor:  float = 2.0,
    exceptions:      Tuple = (Exception,),
    *args,
    **kwargs,
) -> Any:
    """Execute a callable with exponential backoff retry logic."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            last_exc   = e
            sleep_time = backoff_factor ** attempt
            if attempt < max_retries - 1:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s. Retrying in %.1fs",
                    attempt + 1, max_retries, func.__name__, e, sleep_time)
                time.sleep(sleep_time)
            else:
                logger.error(
                    "All %d attempts failed for %s. Last error: %s",
                    max_retries, func.__name__, e)
    raise last_exc


# ==================== SYMBOL HELPERS ====================

def extract_currency_from_symbol(symbol: str) -> Tuple[str, str]:
    """
    Extract base and quote currency from a broker symbol string.
    Only strips suffix when the result is exactly 6 characters.
    """
    clean = symbol
    for suffix in ['.pro', '.raw', '-a', 'm']:
        if clean.endswith(suffix):
            candidate = clean[:-len(suffix)]
            if len(candidate) >= 6:
                clean = candidate
                break
    if clean.startswith('XAU'): return 'XAU', clean[3:]
    if clean.startswith('XAG'): return 'XAG', clean[3:]
    if len(clean) >= 6:         return clean[:3], clean[3:6]
    return clean, 'USD'


def sanitize_filename(filename: str) -> str:
    """Remove characters that are invalid in filenames."""
    for ch in r'<>:"/\|?*':
        filename = filename.replace(ch, '_')
    return filename[:200] if len(filename) > 200 else filename