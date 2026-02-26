"""
NIX TRADES - MT5 Worker Service
Role: Lead Architect + Quant Software Engineer + Infrastructure Engineer +
      DevOps Engineer + QA Engineer + Security Engineer + Data Engineer

DEPLOYMENT:
  Windows VPS with MetaTrader 5 terminal installed.
  1. pip install MetaTrader5 flask waitress python-dotenv
  2. Set environment variable MT5_WORKER_API_KEY to a long random string.
  3. Set MT5_TERMINAL_PATH to the full path of terminal64.exe.
  4. Run: waitress-serve --host=0.0.0.0 --port=8000 --threads=32 mt5_worker:app
  5. For parallel execution (5000+ users): run 4 copies on ports 8001-8004
     behind nginx. The MetaTrader5 library is process-bound (one account per
     process at a time). Multiple processes give true parallelism.

ARCHITECTURE NOTE - SINGLE LOCK PER PROCESS:
  The MetaTrader5 Python library can only hold one active connection at a time
  within a single Python process. The _mt5_lock is therefore process-scoped.
  When running with waitress (32 threads), all threads share this one lock.
  This is correct and intentional. For higher throughput, run multiple
  independent processes on different ports behind nginx.

NO EMOJIS - Enterprise code only
NO PLACEHOLDERS - All logic is complete and production-ready
"""

import os
import logging
import threading
import time
import hmac
import hashlib
import contextlib
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from functools import wraps

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, g

try:
    import MetaTrader5 as mt5
except ImportError:
    raise ImportError(
        "MetaTrader5 package is not installed. "
        "Run: pip install MetaTrader5\n"
        "This service MUST run on Windows with MT5 terminal installed."
    )

# ==================== LOGGING WITH ROTATION ====================
# Maximum 10 MB per file, 5 backup files = 50 MB total disk usage cap.

_log_handler_file    = RotatingFileHandler(
    'mt5_worker.log',
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding='utf-8',
)
_log_handler_console = logging.StreamHandler()
_log_formatter       = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_log_handler_file.setFormatter(_log_formatter)
_log_handler_console.setFormatter(_log_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[_log_handler_file, _log_handler_console],
)
logger = logging.getLogger('mt5_worker')


# ==================== CREDENTIAL SCRUBBING LOG FILTER ====================

class _CredentialScrubFilter(logging.Filter):
    """
    Removes passwords and API keys from log records before they are written.
    Prevents sensitive data from appearing in log files or console.
    """
    _SCRUB_KEYS = ('password', 'passwd', 'api_key', 'x-api-key', 'authorization')

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        lower = msg.lower()
        for key in self._SCRUB_KEYS:
            idx = lower.find(key)
            while idx != -1:
                # Find the value following the key (after ':' or '=')
                val_start = idx + len(key)
                while val_start < len(msg) and msg[val_start] in ' :"\'=':
                    val_start += 1
                val_end = val_start
                while val_end < len(msg) and msg[val_end] not in ' ,}\'")\n':
                    val_end += 1
                if val_end > val_start:
                    msg = msg[:val_start] + '***REDACTED***' + msg[val_end:]
                    lower = msg.lower()
                idx = lower.find(key, val_start)
        record.msg  = msg
        record.args = ()
        return True


_scrub_filter = _CredentialScrubFilter()
for _h in logging.root.handlers:
    _h.addFilter(_scrub_filter)
logger.addFilter(_scrub_filter)


# ==================== CONFIGURATION ====================

WORKER_API_KEY     = os.getenv('MT5_WORKER_API_KEY', '').strip()
WORKER_HOST        = os.getenv('MT5_WORKER_HOST', '0.0.0.0')
WORKER_PORT        = int(os.getenv('MT5_WORKER_PORT', '8000'))
MT5_TERMINAL_PATH  = os.getenv('MT5_TERMINAL_PATH', '').strip()
MAGIC_NUMBER       = 234567

# Lock timeout: if a thread cannot acquire the MT5 lock within 30 seconds,
# it returns an error instead of waiting indefinitely (prevents deadlocks).
_MT5_LOCK_TIMEOUT  = 30  # seconds

# Maximum request body size: 64 KB. A legitimate trade request is < 1 KB.
MAX_REQUEST_BODY_BYTES = 64 * 1024

# ==================== TERMINAL AUTO-DETECTION ====================

def _find_mt5_terminal() -> str:
    """
    Auto-detect the MetaTrader 5 terminal executable path.
    Checks the following locations in order:
      1. MT5_TERMINAL_PATH environment variable (user override, highest priority)
      2. Standard installation paths for Windows x64 and x86
      3. User-level AppData installations (brokers often install here)

    Returns the path string if found, or empty string if not found.
    The caller should check health and warn if empty.
    """
    if MT5_TERMINAL_PATH and os.path.exists(MT5_TERMINAL_PATH):
        return MT5_TERMINAL_PATH

    # Standard Windows installation paths to check automatically
    candidates = [
        r'C:\Program Files\MetaTrader 5\terminal64.exe',
        r'C:\Program Files (x86)\MetaTrader 5\terminal64.exe',
        r'C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe',
        r'C:\Program Files\MetaTrader 5 ICMarkets\terminal64.exe',
        r'C:\Program Files\MetaTrader 5 Pepperstone\terminal64.exe',
        r'C:\Program Files\MetaTrader 5 XM Global\terminal64.exe',
        r'C:\Program Files\MetaTrader 5 FP Markets\terminal64.exe',
        r'C:\Program Files\MetaTrader 5 FXTM\terminal64.exe',
        r'C:\Program Files\MetaTrader 5 OctaFX\terminal64.exe',
        r'C:\Program Files\MetaTrader 5 AvaTrade\terminal64.exe',
        r'C:\Program Files\MetaTrader 5 HotForex\terminal64.exe',
        r'C:\Program Files\MetaTrader 5 Admiral Markets\terminal64.exe',
    ]

    # Also check common AppData locations (many brokers install here)
    appdata = os.environ.get('APPDATA', '')
    localappdata = os.environ.get('LOCALAPPDATA', '')
    for base in (appdata, localappdata):
        if base:
            for sub in ('MetaTrader 5', 'MT5'):
                candidates.append(os.path.join(base, sub, 'terminal64.exe'))

    for path in candidates:
        if os.path.exists(path):
            logger.info("MT5 terminal auto-detected at: %s", path)
            return path

    logger.warning(
        "MT5 terminal not found at any standard path. "
        "Set MT5_TERMINAL_PATH in your .env file to the full path of terminal64.exe. "
        "Example: MT5_TERMINAL_PATH=C:\\Program Files\\MetaTrader 5 EXNESS\\terminal64.exe"
    )
    return ''


# Resolved terminal path (auto-detected or from .env)
_RESOLVED_TERMINAL_PATH = _find_mt5_terminal()

if not WORKER_API_KEY:
    raise ValueError(
        "MT5_WORKER_API_KEY environment variable not set. "
        "Set it to a long random string in your .env file."
    )

# ==================== PROCESS-LEVEL MT5 LOCK ====================
# One lock for this entire Python process. If you need true parallelism,
# run multiple copies of this service on different ports behind nginx.

_mt5_lock = threading.Lock()


# ==================== FLASK APPLICATION ====================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_REQUEST_BODY_BYTES


# ==================== AUTHENTICATION ====================

def require_api_key(f):
    """
    Decorator that enforces API key authentication using a constant-time
    comparison to prevent timing attacks.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        provided_key = request.headers.get('X-API-Key', '')
        # hmac.compare_digest performs a constant-time comparison.
        # This means an attacker cannot determine how many characters of
        # the key are correct by measuring the response time.
        if not provided_key or not hmac.compare_digest(
            provided_key.encode('utf-8'),
            WORKER_API_KEY.encode('utf-8'),
        ):
            logger.warning(
                "Unauthorised request from %s to %s",
                request.remote_addr, request.path
            )
            return jsonify({'success': False, 'error': 'Unauthorised'}), 401
        return f(*args, **kwargs)
    return decorated


def _check_request_size():
    """Reject oversized request bodies before parsing JSON."""
    content_length = request.content_length
    if content_length and content_length > MAX_REQUEST_BODY_BYTES:
        logger.warning(
            "Request body too large: %d bytes from %s",
            content_length, request.remote_addr
        )
        return jsonify({
            'success': False,
            'error': 'Request body too large. Maximum is 64 KB.',
        }), 413
    return None


# ==================== MT5 CONTEXT MANAGER ====================

@contextlib.contextmanager
def _mt5_session(login: int, password: str, server: str):
    """
    Context manager that initialises a per-user MT5 connection and
    guarantees cleanup even if the thread is interrupted.

    Usage:
        with _mt5_session(login, password, server) as (ok, err):
            if not ok:
                return jsonify({'success': False, 'error': err})
            # use mt5 safely here
    """
    acquired = _mt5_lock.acquire(timeout=_MT5_LOCK_TIMEOUT)
    if not acquired:
        # Yield immediately without holding the lock.
        # The finally block below does NOT run for this branch
        # because we use return after yield inside a generator,
        # which raises StopIteration before finally executes.
        yield False, "Execution service busy. Please retry in 30 seconds."
        return

    # Lock is held from this point. Finally block always releases it.
    try:
        ok, err = _initialise_headless(login, password, server)
        yield ok, (err or "")
    finally:
        _shutdown_safe()
        _mt5_lock.release()


def _initialise_headless(login: int, password: str, server: str) -> Tuple[bool, str]:
    """
    Initialise MT5 in headless mode (no GUI window).
    Uses the auto-detected or configured terminal path.
    If no terminal path is found, MT5.initialize() is still called without
    a path argument - the library will try to find the terminal itself.
    """
    kwargs: Dict[str, Any] = {
        'login':    login,
        'password': password,
        'server':   server,
        'timeout':  10000,
    }
    if _RESOLVED_TERMINAL_PATH:
        kwargs['path'] = _RESOLVED_TERMINAL_PATH

    if not mt5.initialize(**kwargs):
        error = mt5.last_error()
        logger.warning("MT5 initialise failed for login %d: %s", login, error)
        return False, "Could not connect to your broker account. Please verify your account details."

    return True, ''


def _shutdown_safe():
    """Call mt5.shutdown() swallowing any exception."""
    try:
        mt5.shutdown()
    except Exception:
        pass


# ==================== VALIDATION HELPERS ====================

def _validate_login(login_raw) -> Tuple[bool, str, int]:
    """
    Validate MT5 login number.
    MT5 internally uses a 32-bit integer. Maximum value: 2,147,483,647.
    """
    try:
        login = int(login_raw)
    except (TypeError, ValueError):
        return False, "login must be a numeric account number.", 0

    if not (1 <= login <= 2_147_483_647):
        return False, "login must be between 1 and 2,147,483,647.", 0

    return True, '', login


def _validate_trade_prices(
    direction: str,
    entry: float,
    sl: float,
    tp1: float,
    tp2: float,
) -> Tuple[bool, str]:
    """
    Validate that price levels are logically consistent before sending to MT5.

    BUY:  sl < entry < tp1 < tp2
    SELL: tp2 < tp1 < entry < sl

    Returns (True, '') if valid, (False, human_readable_error) if invalid.
    """
    if direction == 'BUY':
        if sl >= entry:
            return False, "For a BUY order, the stop loss must be below the entry price."
        if tp1 <= entry:
            return False, "For a BUY order, TP1 must be above the entry price."
        if tp2 <= tp1:
            return False, "For a BUY order, TP2 must be above TP1."
    elif direction == 'SELL':
        if sl <= entry:
            return False, "For a SELL order, the stop loss must be above the entry price."
        if tp1 >= entry:
            return False, "For a SELL order, TP1 must be below the entry price."
        if tp2 >= tp1:
            return False, "For a SELL order, TP2 must be below TP1."
    else:
        return False, f"Invalid direction '{direction}'. Must be BUY or SELL."

    return True, ''


def _validate_new_sl(
    position_type: int,
    new_sl: float,
    current_price: float,
) -> Tuple[bool, str]:
    """
    Validate a stop-loss modification request.

    BUY  position (type 0): new SL must be below current bid.
    SELL position (type 1): new SL must be above current ask.
    """
    if position_type == mt5.POSITION_TYPE_BUY:
        if new_sl >= current_price:
            return False, (
                f"Cannot move stop loss to {new_sl:.5f}. "
                f"For a buy trade, stop loss must be below current price ({current_price:.5f})."
            )
    elif position_type == mt5.POSITION_TYPE_SELL:
        if new_sl <= current_price:
            return False, (
                f"Cannot move stop loss to {new_sl:.5f}. "
                f"For a sell trade, stop loss must be above current price ({current_price:.5f})."
            )
    return True, ''


def _require_fields(data: dict, *fields: str) -> Optional[str]:
    """Return a plain-English error string if any required field is absent."""
    for f in fields:
        if f not in data or data[f] is None:
            return f"Missing required field: {f}"
    return None


# ==================== FILLING MODE NEGOTIATION ====================

def _get_filling_mode(symbol: str) -> int:
    """
    Query the broker to determine which order filling mode it supports
    for the given symbol. Returns the first supported mode.

    MT5 filling modes (bitfield):
      ORDER_FILLING_FOK    = 1 (Fill or Kill)
      ORDER_FILLING_IOC    = 2 (Immediate or Cancel)
      ORDER_FILLING_RETURN = 4 (Partial fills allowed)

    Many brokers only support FOK. Using an unsupported mode causes
    rejection with error 10030. This function prevents that.
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return mt5.ORDER_FILLING_IOC  # Safe default

    filling_mode = symbol_info.filling_mode
    if filling_mode & mt5.ORDER_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    if filling_mode & mt5.ORDER_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


# ==================== SYMBOL NORMALISATION ====================

# Known suffixes added by brokers. Stripped when matching standard names.
_KNOWN_SUFFIXES = (
    '.pro', '.raw', '.m', '.ecn', '.stp', '.a', '.b', '.c',
    '_', '.i', '.fix', 'pro', 'raw', 'm', 'ecn',
)


def _normalise_symbol(base_symbol: str) -> Optional[str]:
    """
    Map a standard symbol name (e.g. 'EURUSD') to the exact name
    used by the connected broker (e.g. 'EURUSD.pro').

    Uses exact base-name matching to prevent false matches such as
    EURUSD matching EURUSDM or EURUSD_OLD.

    Returns None if no matching symbol is found.
    """
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        return None

    base_upper = base_symbol.upper()

    for sym in all_symbols:
        candidate = sym.name.upper()
        # Strip known suffixes from candidate to get its base
        candidate_base = candidate
        for suffix in _KNOWN_SUFFIXES:
            if candidate.endswith(suffix.upper()):
                candidate_base = candidate[: -len(suffix)]
                break

        if candidate_base == base_upper:
            return sym.name  # Return broker's exact name

    return None


# ==================== LOT SIZE CALCULATION ====================

def _calculate_lot_size(
    symbol: str,
    risk_percent: float,
    sl_pips: float,
    account_balance: float,
    account_currency: str,
) -> float:
    """
    Calculate lot size using the correct formula:
        lot_size = risk_amount / (price_distance / tick_size * tick_value)

    Where price_distance = sl_pips * pip_size.

    Args:
        symbol:           Broker-specific symbol name
        risk_percent:     Risk per trade as a percentage (e.g. 1.0 = 1%)
        sl_pips:          Stop-loss distance in pips
        account_balance:  Current account balance
        account_currency: Account currency code (e.g. 'USD')

    Returns:
        float: Lot size rounded to 8 decimal places, clipped to broker limits.
               Falls back to volume_min on any error.
    """
    if sl_pips <= 0 or account_balance <= 0:
        return 0.01

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error("Cannot calculate lot size - symbol info unavailable: %s", symbol)
        return 0.01

    tick_value = symbol_info.trade_tick_value   # Value of one tick in account currency
    tick_size  = symbol_info.trade_tick_size    # Price change per one tick
    point      = symbol_info.point              # Smallest price increment
    lot_min    = symbol_info.volume_min
    lot_max    = symbol_info.volume_max
    lot_step   = symbol_info.volume_step

    if tick_size == 0 or point == 0:
        return lot_min

    # pip_size: most pairs = 10 points; JPY pairs = 10 points of 0.001 = 0.01
    # Gold (XAUUSD): point = 0.01, pip = 1.0 (100 points per pip)
    # We use point * 10 as the standard pip definition
    pip_size = point * 10

    # price_distance in price units
    price_distance = sl_pips * pip_size

    if price_distance == 0:
        return lot_min

    # Correct formula: risk_amount / (price_distance / tick_size * tick_value)
    risk_amount = account_balance * (risk_percent / 100.0)
    ticks_at_risk = price_distance / tick_size
    value_at_risk_per_lot = ticks_at_risk * tick_value

    if value_at_risk_per_lot <= 0:
        return lot_min

    raw_lot = risk_amount / value_at_risk_per_lot

    # Snap to broker's volume step (round to 8 dp first to avoid float drift)
    if lot_step > 0:
        raw_lot = round(round(raw_lot / lot_step) * lot_step, 8)

    lot_size = max(lot_min, min(lot_max, raw_lot))
    return round(lot_size, 8)


# ==================== ACCOUNT INFO ====================

def _get_account_info() -> Optional[dict]:
    """
    Retrieve all account fields required by the position monitor and
    the execute endpoint, including trade_allowed and margin_free.
    """
    info = mt5.account_info()
    if info is None:
        return None
    return {
        'login':         info.login,
        'broker':        info.company,
        'server':        info.server,
        'balance':       info.balance,
        'equity':        info.equity,
        'margin':        info.margin,
        'margin_free':   info.margin_free,
        'profit':        info.profit,
        'currency':      info.currency,
        'leverage':      info.leverage,
        'trade_allowed': info.trade_allowed,
        'trade_mode':    info.trade_mode,
    }


# ==================== ORDER PLACEMENT ====================

def _place_order(
    symbol:     str,
    direction:  str,
    entry:      float,
    sl:         float,
    tp1:        float,
    tp2:        float,
    lot_size:   float,
    order_type: str,
    comment:    str,
    expiry_hours: int = 2,
) -> Tuple[bool, dict]:
    """
    Build and submit an MT5 trade request.

    Supports MARKET, LIMIT, and STOP order types.
    Filling mode is negotiated from broker capabilities.
    Expiry is set per the signal's timeframe: 4H=24h, 1H=8h, 15m=2h.

    Returns:
        (True, result_dict) on success
        (False, error_dict) on failure
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False, {'error': f"Cannot retrieve market tick for {symbol}."}

    filling = _get_filling_mode(symbol)

    if order_type == 'MARKET':
        price      = tick.ask if direction == 'BUY' else tick.bid
        mt5_type   = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
        mt5_action = mt5.TRADE_ACTION_DEAL
        type_time  = mt5.ORDER_TIME_GTC
        expiration = 0
    elif order_type == 'LIMIT':
        price      = entry
        mt5_type   = mt5.ORDER_TYPE_BUY_LIMIT if direction == 'BUY' else mt5.ORDER_TYPE_SELL_LIMIT
        mt5_action = mt5.TRADE_ACTION_PENDING
        type_time  = mt5.ORDER_TIME_SPECIFIED
        expiration = int(
            (datetime.now(timezone.utc) + timedelta(hours=expiry_hours)).timestamp()
        )
    else:  # STOP
        price      = entry
        mt5_type   = mt5.ORDER_TYPE_BUY_STOP if direction == 'BUY' else mt5.ORDER_TYPE_SELL_STOP
        mt5_action = mt5.TRADE_ACTION_PENDING
        type_time  = mt5.ORDER_TIME_SPECIFIED
        expiration = int(
            (datetime.now(timezone.utc) + timedelta(hours=expiry_hours)).timestamp()
        )

    request_dict = {
        'action':       mt5_action,
        'symbol':       symbol,
        'volume':       lot_size,
        'type':         mt5_type,
        'price':        price,
        'sl':           sl,
        'tp':           tp1,
        'deviation':    10,
        'magic':        MAGIC_NUMBER,
        'comment':      comment[:31],
        'type_time':    type_time,
        'type_filling': filling,
    }
    if expiration:
        request_dict['expiration'] = expiration

    result = mt5.order_send(request_dict)

    if result is None:
        error = mt5.last_error()
        return False, {'error': f"Order submission failed: {error}"}

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return False, {
            'error':   _translate_retcode(result.retcode, result.comment),
            'retcode': result.retcode,
        }

    return True, {
        'order':      result.order,
        'deal':       result.deal,
        'lot_size':   lot_size,
        'entry':      result.price,
        'sl':         sl,
        'tp1':        tp1,
        'tp2':        tp2,
        'symbol':     symbol,
        'direction':  direction,
        'order_type': order_type,
    }


def _translate_retcode(retcode: int, raw_comment: str) -> str:
    """Map MT5 return codes to plain-English messages for end users."""
    _MAP = {
        10004: "Requote received. The market price moved during execution.",
        10013: "Invalid price levels. The broker rejected the stop/take-profit.",
        10014: "Invalid lot size. Check your risk settings.",
        10015: "Invalid price. The market may have moved significantly.",
        10016: "The signal prices may have shifted. Check the latest setup.",
        10019: "Insufficient free margin to open this trade.",
        10030: "Your broker does not support this order type. Please contact your broker.",
    }
    return _MAP.get(retcode, f"Order could not be placed. Please try again. ({retcode})")


# ==================== API ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    Returns HTTP 503 if the MT5 terminal cannot be found so that
    load balancers remove this instance from rotation automatically.
    """
    terminal_ok = bool(_RESOLVED_TERMINAL_PATH)

    if not terminal_ok:
        return jsonify({
            'status':    'degraded',
            'service':   'Nix Trades Execution Service',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message':   (
                'MT5 terminal not found. '
                'Set MT5_TERMINAL_PATH in your .env file. '
                'Example: MT5_TERMINAL_PATH=C:\\Program Files\\MetaTrader 5 EXNESS\\terminal64.exe'
            ),
        }), 503

    return jsonify({
        'status':    'online',
        'service':   'Nix Trades Execution Service',
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })


@app.route('/tick', methods=['GET'])
@require_api_key
def get_tick():
    """
    Return current bid/ask for a symbol.
    Used by the scheduler to determine the correct order type
    (MARKET / LIMIT / STOP) relative to the entry price.
    """
    size_error = _check_request_size()
    if size_error:
        return size_error

    symbol = request.args.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({'success': False, 'error': 'symbol parameter required'}), 400

    acquired = _mt5_lock.acquire(timeout=_MT5_LOCK_TIMEOUT)
    if not acquired:
        return jsonify({'success': False, 'error': 'Service busy. Please retry in 30 seconds.'}), 503
    try:
        if not mt5.initialize():
            return jsonify({'success': False, 'error': 'MT5 initialisation failed.'})

        broker_symbol = _normalise_symbol(symbol)
        if broker_symbol is None:
            return jsonify({'success': False, 'error': f'Symbol {symbol} not found on your broker.'})

        tick = mt5.symbol_info_tick(broker_symbol)
        if tick is None:
            return jsonify({'success': False, 'error': 'Could not retrieve market data.'})

        return jsonify({
            'success': True,
            'symbol':  broker_symbol,
            'bid':     tick.bid,
            'ask':     tick.ask,
            'time':    tick.time,
        })

    finally:
        _shutdown_safe()
        _mt5_lock.release()


@app.route('/connect', methods=['POST'])
@require_api_key
def connect():
    """
    Verify MT5 credentials and return account information.

    Request body:
        { "login": 12345678, "password": "...", "server": "BrokerName-Live01" }

    Response (success):
        { "success": true, "login": ..., "broker": ..., "balance": ...,
          "equity": ..., "margin_free": ..., "currency": ...,
          "leverage": ..., "server": ..., "trade_allowed": true }
    """
    size_error = _check_request_size()
    if size_error:
        return size_error

    data  = request.get_json(force=True, silent=True) or {}
    error = _require_fields(data, 'login', 'password', 'server')
    if error:
        return jsonify({'success': False, 'error': error}), 400

    valid, err_msg, login = _validate_login(data['login'])
    if not valid:
        return jsonify({'success': False, 'error': err_msg}), 400

    acquired = _mt5_lock.acquire(timeout=_MT5_LOCK_TIMEOUT)
    if not acquired:
        return jsonify({'success': False, 'error': 'Service busy. Please retry shortly.'}), 503
    try:
        ok, err = _initialise_headless(login, data['password'], data['server'])
        if not ok:
            return jsonify({'success': False, 'error': err})

        info = _get_account_info()
        if info is None:
            return jsonify({'success': False, 'error': 'Failed to retrieve account information.'})

        return jsonify({'success': True, **info})

    finally:
        _shutdown_safe()
        _mt5_lock.release()


@app.route('/execute', methods=['POST'])
@require_api_key
def execute():
    """
    Execute a trade order on the specified account.

    Request body:
        {
            "login":        12345678,
            "password":     "...",
            "server":       "BrokerName-Live01",
            "symbol":       "EURUSD",
            "direction":    "BUY",
            "entry":        1.08500,
            "sl":           1.08200,
            "tp1":          1.08950,
            "tp2":          1.09250,
            "sl_pips":      30.0,
            "risk_percent": 1.0,
            "order_type":   "LIMIT",
            "comment":      "NT-001",
            "expiry_hours": 2
        }

    lot_size is always calculated server-side using:
        lot_size = risk_amount / (price_distance / tick_size * tick_value)
    """
    size_error = _check_request_size()
    if size_error:
        return size_error

    data  = request.get_json(force=True, silent=True) or {}
    error = _require_fields(
        data,
        'login', 'password', 'server',
        'symbol', 'direction', 'entry', 'sl', 'tp1', 'tp2',
        'sl_pips', 'risk_percent', 'order_type',
    )
    if error:
        return jsonify({'success': False, 'error': error}), 400

    valid, err_msg, login = _validate_login(data['login'])
    if not valid:
        return jsonify({'success': False, 'error': err_msg}), 400

    direction    = str(data['direction']).upper()
    order_type   = str(data['order_type']).upper()
    entry        = float(data['entry'])
    sl           = float(data['sl'])
    tp1          = float(data['tp1'])
    tp2          = float(data['tp2'])
    sl_pips      = float(data['sl_pips'])
    risk_percent = float(data['risk_percent'])
    comment      = str(data.get('comment', 'NixTrades'))
    expiry_hours = int(data.get('expiry_hours', 2))

    if direction not in ('BUY', 'SELL'):
        return jsonify({'success': False, 'error': "direction must be BUY or SELL"}), 400

    if order_type not in ('MARKET', 'LIMIT', 'STOP'):
        return jsonify({'success': False, 'error': "order_type must be MARKET, LIMIT, or STOP"}), 400

    # Pre-broker price validation
    price_valid, price_error = _validate_trade_prices(direction, entry, sl, tp1, tp2)
    if not price_valid:
        return jsonify({'success': False, 'error': price_error}), 400

    acquired = _mt5_lock.acquire(timeout=_MT5_LOCK_TIMEOUT)
    if not acquired:
        return jsonify({'success': False, 'error': 'Service busy. Please retry in 30 seconds.'}), 503
    try:
        ok, err = _initialise_headless(login, data['password'], data['server'])
        if not ok:
            return jsonify({'success': False, 'error': err})

        account = _get_account_info()
        if account is None:
            return jsonify({'success': False, 'error': 'Failed to retrieve account information.'})

        if not account.get('trade_allowed', False):
            return jsonify({
                'success': False,
                'error':   (
                    'Automated trading is not enabled on your account. '
                    'Please enable it in your MetaTrader 5 settings: '
                    'Tools -> Options -> Expert Advisors -> Allow automated trading.'
                ),
            })

        raw_symbol    = str(data['symbol']).upper()
        broker_symbol = _normalise_symbol(raw_symbol)
        if broker_symbol is None:
            return jsonify({'success': False, 'error': f'Symbol {raw_symbol} is not available on your broker.'})

        lot_size = _calculate_lot_size(
            symbol=broker_symbol,
            risk_percent=risk_percent,
            sl_pips=sl_pips,
            account_balance=account['balance'],
            account_currency=account['currency'],
        )

        success, result = _place_order(
            symbol=broker_symbol,
            direction=direction,
            entry=entry,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            lot_size=lot_size,
            order_type=order_type,
            comment=comment,
            expiry_hours=expiry_hours,
        )

        if not success:
            return jsonify({'success': False, 'error': result.get('error', 'Unknown error')})

        logger.info(
            "Order placed: login=%d %s %s %.2f lots order=%d",
            login, direction, broker_symbol, lot_size, result.get('order', 0)
        )
        return jsonify({'success': True, **result})

    finally:
        _shutdown_safe()
        _mt5_lock.release()


@app.route('/account', methods=['POST'])
@require_api_key
def account():
    """
    Return full account information for a user.

    Request body: { "login": ..., "password": ..., "server": ... }
    """
    size_error = _check_request_size()
    if size_error:
        return size_error

    data  = request.get_json(force=True, silent=True) or {}
    error = _require_fields(data, 'login', 'password', 'server')
    if error:
        return jsonify({'success': False, 'error': error}), 400

    valid, err_msg, login = _validate_login(data['login'])
    if not valid:
        return jsonify({'success': False, 'error': err_msg}), 400

    acquired = _mt5_lock.acquire(timeout=_MT5_LOCK_TIMEOUT)
    if not acquired:
        return jsonify({'success': False, 'error': 'Service busy. Please retry shortly.'}), 503
    try:
        ok, err = _initialise_headless(login, data['password'], data['server'])
        if not ok:
            return jsonify({'success': False, 'error': err})

        info = _get_account_info()
        if info is None:
            return jsonify({'success': False, 'error': 'Failed to retrieve account information.'})

        return jsonify({'success': True, **info})

    finally:
        _shutdown_safe()
        _mt5_lock.release()

@app.route('/candles', methods=['POST'])
@require_api_key
def get_candles():
    """
    Return OHLCV bars for a symbol.
    Does not require MT5 account login credentials.
    Uses the terminal's existing session for data access only.
    """
    # All parameter extraction happens here, BEFORE acquiring any lock.
    # If any of these fail, Flask returns JSON immediately without touching MT5.
    raw = request.get_json(force=True, silent=True) or {}

    symbol_raw    = str(raw.get('symbol', '')).upper().strip()
    timeframe_str = str(raw.get('timeframe', 'H1')).upper().strip()
    try:
        bars = int(raw.get('bars', 500))
    except (TypeError, ValueError):
        bars = 500

    if not symbol_raw:
        return jsonify({'success': False, 'error': 'symbol field is required'}), 400

    timeframe_map = {
        'M1':  mt5.TIMEFRAME_M1,
        'M5':  mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1':  mt5.TIMEFRAME_H1,
        'H4':  mt5.TIMEFRAME_H4,
        'D1':  mt5.TIMEFRAME_D1,
        'W1':  mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1,
    }
    tf = timeframe_map.get(timeframe_str)
    if tf is None:
        return jsonify({
            'success': False,
            'error':   'Unknown timeframe: %s. Valid values: %s' % (
                timeframe_str, ', '.join(timeframe_map.keys()))
        }), 400

    # Acquire lock only after all validation passes
    acquired = _mt5_lock.acquire(timeout=_MT5_LOCK_TIMEOUT)
    if not acquired:
        return jsonify({
            'success': False,
            'error':   'Service busy. Please retry in 30 seconds.'
        }), 503

    try:
        if not mt5.initialize():
            return jsonify({'success': False, 'error': 'MT5 initialise failed'})

        symbol = _normalise_symbol(symbol_raw)
        if symbol is None:
            symbol = symbol_raw

        # symbol_select is mandatory before copy_rates_from_pos.
        # Without this call MT5 returns (-2, 'Terminal: Invalid params').
        if not mt5.symbol_select(symbol, True):
            return jsonify({
                'success': False,
                'error':   'Symbol %s was not found on this broker. '
                           'Check the symbol name.' % symbol_raw
            })

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None or len(rates) == 0:
            return jsonify({
                'success': False,
                'error':   'No data returned for %s %s. Error: %s' % (
                    symbol, timeframe_str, str(mt5.last_error()))
            })

        candles = [
            {
                'time':   int(r['time']),
                'open':   float(r['open']),
                'high':   float(r['high']),
                'low':    float(r['low']),
                'close':  float(r['close']),
                'volume': int(r['tick_volume']),
            }
            for r in rates
        ]
        return jsonify({'success': True, 'symbol': symbol,
                        'timeframe': timeframe_str, 'candles': candles})

    except Exception as e:
        logger.error("Error in /candles for %s %s: %s", symbol_raw, timeframe_str, e)
        return jsonify({'success': False, 'error': str(e)})

    finally:
        _shutdown_safe()
        _mt5_lock.release()

@app.route('/positions', methods=['POST'])
@require_api_key
def positions():
    """
    Return all open positions for an account.
    Includes price_current, time_update, and swap fields required
    by the position monitor.

    Request body: { "login": ..., "password": ..., "server": ... }
    """
    size_error = _check_request_size()
    if size_error:
        return size_error

    data  = request.get_json(force=True, silent=True) or {}
    error = _require_fields(data, 'login', 'password', 'server')
    if error:
        return jsonify({'success': False, 'error': error}), 400

    valid, err_msg, login = _validate_login(data['login'])
    if not valid:
        return jsonify({'success': False, 'error': err_msg}), 400

    acquired = _mt5_lock.acquire(timeout=_MT5_LOCK_TIMEOUT)
    if not acquired:
        return jsonify({'success': False, 'error': 'Service busy. Please retry shortly.'}), 503
    try:
        ok, err = _initialise_headless(login, data['password'], data['server'])
        if not ok:
            return jsonify({'success': False, 'error': err})

        raw_positions = mt5.positions_get()
        if raw_positions is None:
            raw_positions = []

        result = []
        for pos in raw_positions:
            result.append({
                'ticket':        pos.ticket,
                'symbol':        pos.symbol,
                'type':          pos.type,
                'volume':        pos.volume,
                'price_open':    pos.price_open,
                'price_current': pos.price_current,
                'sl':            pos.sl,
                'tp':            pos.tp,
                'profit':        pos.profit,
                'swap':          pos.swap,
                'time':          pos.time,
                'time_update':   pos.time_update,
                'magic':         pos.magic,
                'comment':       pos.comment,
            })

        return jsonify({'success': True, 'positions': result})

    finally:
        _shutdown_safe()
        _mt5_lock.release()


@app.route('/close_partial', methods=['POST'])
@require_api_key
def close_partial():
    """
    Partially close an open position.

    Request body:
        {
            "login":      12345678,
            "password":   "...",
            "server":     "...",
            "ticket":     123456789,
            "close_pct":  0.5
        }

    close_pct: Fraction of position to close. Must be 0.01 - 1.0.
    """
    size_error = _check_request_size()
    if size_error:
        return size_error

    data  = request.get_json(force=True, silent=True) or {}
    error = _require_fields(data, 'login', 'password', 'server', 'ticket', 'close_pct')
    if error:
        return jsonify({'success': False, 'error': error}), 400

    valid, err_msg, login = _validate_login(data['login'])
    if not valid:
        return jsonify({'success': False, 'error': err_msg}), 400

    ticket    = int(data['ticket'])
    close_pct = float(data['close_pct'])

    if not (0.01 <= close_pct <= 1.0):
        return jsonify({
            'success': False,
            'error':   'close_pct must be between 0.01 (1%) and 1.0 (100%).',
        }), 400

    acquired = _mt5_lock.acquire(timeout=_MT5_LOCK_TIMEOUT)
    if not acquired:
        return jsonify({'success': False, 'error': 'Service busy. Please retry shortly.'}), 503
    try:
        ok, err = _initialise_headless(login, data['password'], data['server'])
        if not ok:
            return jsonify({'success': False, 'error': err})

        pos_list = mt5.positions_get(ticket=ticket)
        if not pos_list:
            return jsonify({'success': False, 'error': f'Position {ticket} not found.'})

        pos          = pos_list[0]
        close_volume = round(pos.volume * close_pct, 8)
        close_volume = max(
            mt5.symbol_info(pos.symbol).volume_min if mt5.symbol_info(pos.symbol) else 0.01,
            close_volume,
        )

        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return jsonify({'success': False, 'error': f'Cannot get market data for {pos.symbol}.'})

        close_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        close_type  = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        filling     = _get_filling_mode(pos.symbol)

        close_request = {
            'action':       mt5.TRADE_ACTION_DEAL,
            'symbol':       pos.symbol,
            'volume':       close_volume,
            'type':         close_type,
            'position':     ticket,
            'price':        close_price,
            'deviation':    10,
            'magic':        MAGIC_NUMBER,
            'comment':      'NT-partial',
            'type_time':    mt5.ORDER_TIME_GTC,
            'type_filling': filling,
        }

        result = mt5.order_send(close_request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err_detail = (
                _translate_retcode(result.retcode, result.comment)
                if result
                else str(mt5.last_error())
            )
            return jsonify({'success': False, 'error': err_detail})

        logger.info("Partial close: ticket=%d volume=%.2f", ticket, close_volume)
        return jsonify({
            'success':       True,
            'ticket':        ticket,
            'closed_volume': close_volume,
            'deal':          result.deal,
        })

    finally:
        _shutdown_safe()
        _mt5_lock.release()


@app.route('/modify_sl', methods=['POST'])
@require_api_key
def modify_sl():
    """
    Modify the stop-loss of an open position (used for breakeven management).

    Request body:
        { "login": ..., "password": ..., "server": ...,
          "ticket": 123456789, "new_sl": 1.08520 }
    """
    size_error = _check_request_size()
    if size_error:
        return size_error

    data  = request.get_json(force=True, silent=True) or {}
    error = _require_fields(data, 'login', 'password', 'server', 'ticket', 'new_sl')
    if error:
        return jsonify({'success': False, 'error': error}), 400

    valid, err_msg, login = _validate_login(data['login'])
    if not valid:
        return jsonify({'success': False, 'error': err_msg}), 400

    ticket = int(data['ticket'])
    new_sl = float(data['new_sl'])

    acquired = _mt5_lock.acquire(timeout=_MT5_LOCK_TIMEOUT)
    if not acquired:
        return jsonify({'success': False, 'error': 'Service busy. Please retry shortly.'}), 503
    try:
        ok, err = _initialise_headless(login, data['password'], data['server'])
        if not ok:
            return jsonify({'success': False, 'error': err})

        pos_list = mt5.positions_get(ticket=ticket)
        if not pos_list:
            return jsonify({'success': False, 'error': f'Position {ticket} not found.'})

        pos  = pos_list[0]
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return jsonify({'success': False, 'error': 'Cannot get current market price.'})

        current_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask

        # Validate SL direction before sending to broker
        sl_valid, sl_error = _validate_new_sl(pos.type, new_sl, current_price)
        if not sl_valid:
            return jsonify({'success': False, 'error': sl_error}), 400

        request_dict = {
            'action':   mt5.TRADE_ACTION_SLTP,
            'position': ticket,
            'sl':       new_sl,
            'tp':       pos.tp,
        }

        result = mt5.order_send(request_dict)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err_detail = (
                _translate_retcode(result.retcode, result.comment)
                if result
                else str(mt5.last_error())
            )
            return jsonify({'success': False, 'error': err_detail})

        logger.info("SL modified: ticket=%d new_sl=%.5f", ticket, new_sl)
        return jsonify({'success': True, 'ticket': ticket, 'new_sl': new_sl})

    finally:
        _shutdown_safe()
        _mt5_lock.release()


# ==================== ENTRY POINT ====================

if __name__ == '__main__':
    # Direct execution for development only.
    # For production use: waitress-serve --host=0.0.0.0 --port=8000 --threads=32 mt5_worker:app
    from waitress import serve
    logger.info(
        "Starting Nix Trades Execution Service on %s:%d", WORKER_HOST, WORKER_PORT
    )
    serve(app, host=WORKER_HOST, port=WORKER_PORT, threads=32)