"""
NIX TRADES - MT5 Worker Service
Runs on a Windows VPS/machine. Exposes a REST API that the Telegram bot
calls to execute trades on behalf of multiple users simultaneously.

Role: Lead Architect + Quant Software Engineer + Infrastructure Engineer
Fixes:
  - Root cause fix: MetaTrader5 Python library is Windows-only and requires the
    MT5 terminal installed. This service runs ON WINDOWS and accepts trade
    instructions from the Linux/cloud bot via HTTP.
  - Per-user connection management: Each user gets their own isolated MT5 session.
  - Headless operation: MT5 terminal is launched programmatically, not by the user.
  - API key authentication to prevent unauthorised access.
  - Lot size calculation integrated here for accuracy against live account data.

DEPLOYMENT:
  1. Install Python 3.11 on Windows VPS.
  2. Install MT5 terminal from your broker.
  3. pip install MetaTrader5 flask flask-limiter pywin32
  4. Set environment variables: MT5_WORKER_API_KEY
  5. Run: python mt5_worker.py
  6. Set MT5_WORKER_URL in your bot .env to this machine's IP:port.

NO EMOJIS - Professional code only
"""

import os
import logging
import threading
import time
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from functools import wraps

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify

try:
    import MetaTrader5 as mt5
except ImportError:
    raise ImportError(
        "MetaTrader5 package is not installed. "
        "Run: pip install MetaTrader5\n"
        "This service MUST run on Windows with MT5 terminal installed."
    )

# ==================== LOGGING ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mt5_worker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('mt5_worker')

# ==================== CONFIGURATION ====================

WORKER_API_KEY = os.getenv('MT5_WORKER_API_KEY', '')
WORKER_HOST = os.getenv('MT5_WORKER_HOST', '0.0.0.0')
WORKER_PORT = int(os.getenv('MT5_WORKER_PORT', '8000'))
MAGIC_NUMBER = 234567  # Unique identifier for bot-placed orders
MAX_SESSIONS = 50      # Maximum concurrent user MT5 sessions
SESSION_TIMEOUT_SECONDS = 300  # Idle session timeout

if not WORKER_API_KEY:
    raise ValueError(
        "MT5_WORKER_API_KEY environment variable not set. "
        "Set it to a long random string and configure the same value in your bot .env."
    )

# ==================== SESSION MANAGER ====================

class UserSession:
    """Represents one user's MT5 connection state."""

    def __init__(self, telegram_id: int):
        self.telegram_id = telegram_id
        self.connected = True
        self.last_active = time.time()
        self.lock = threading.Lock()

    def touch(self):
        self.last_active = time.time()

    def is_expired(self) -> bool:
        return time.time() - self.last_active > SESSION_TIMEOUT_SECONDS


class SessionManager:
    """Thread-safe manager for per-user MT5 sessions."""

    def __init__(self):
        self._sessions: Dict[int, UserSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, telegram_id: int) -> UserSession:
        with self._lock:
            if telegram_id not in self._sessions:
                if len(self._sessions) >= MAX_SESSIONS:
                    self._evict_oldest()
                self._sessions[telegram_id] = UserSession(telegram_id)
            session = self._sessions[telegram_id]
            session.touch()
            return session

    def remove(self, telegram_id: int):
        with self._lock:
            self._sessions.pop(telegram_id, None)

    def _evict_oldest(self):
        """Remove the session that has been idle the longest."""
        if not self._sessions:
            return
        oldest_id = min(self._sessions, key=lambda tid: self._sessions[tid].last_active)
        logger.info("Evicting idle session for user %d", oldest_id)
        self._sessions.pop(oldest_id)

    def cleanup_expired(self):
        """Remove sessions that have exceeded their idle timeout."""
        with self._lock:
            expired = [tid for tid, s in self._sessions.items() if s.is_expired()]
            for tid in expired:
                logger.info("Session expired for user %d", tid)
                self._sessions.pop(tid)


session_manager = SessionManager()

# ==================== FLASK APP ====================

app = Flask(__name__)


def require_api_key(f):
    """Decorator to enforce API key authentication on all routes."""
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key', '')
        if not key or key != WORKER_API_KEY:
            logger.warning(
                "Unauthorised request from %s to %s",
                request.remote_addr, request.path
            )
            return jsonify({'success': False, 'error': 'Unauthorised'}), 401
        return f(*args, **kwargs)
    return decorated


def _connect_mt5(login: int, password: str, server: str) -> Tuple[bool, str]:
    """
    Initialise and log in to MT5 terminal.
    MT5 is launched headlessly if not already running.

    Args:
        login:    MT5 account login number
        password: MT5 account password
        server:   MT5 broker server name

    Returns:
        Tuple[bool, str]: (success, message)
    """
    for attempt in range(3):
        try:
            if not mt5.initialize():
                error = mt5.last_error()
                logger.error("MT5 initialize() failed (attempt %d): %s", attempt + 1, error)
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                return False, f"MT5 terminal could not be started: {error}"

            if not mt5.login(login=login, password=password, server=server):
                error = mt5.last_error()
                mt5.shutdown()
                logger.error("MT5 login failed (attempt %d): %s", attempt + 1, error)
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                return False, f"Login failed: {error}"

            account_info = mt5.account_info()
            if account_info is None:
                mt5.shutdown()
                return False, "Could not retrieve account information after login."

            logger.info(
                "MT5 connected: login=%d, broker=%s, balance=%.2f %s",
                login, account_info.company, account_info.balance, account_info.currency
            )
            return True, "Connected"

        except Exception as e:
            logger.error("MT5 connect exception (attempt %d): %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    return False, "Connection failed after 3 attempts"


def _calculate_lot_size(
    balance: float,
    risk_percent: float,
    sl_pips: float,
    symbol: str
) -> float:
    """
    Calculate lot size using fixed fractional position sizing.
    Risk amount = balance * risk_percent / 100
    Lot size    = risk_amount / (sl_pips * pip_value_per_lot)

    Args:
        balance:      Account balance
        risk_percent: Risk per trade (e.g., 1.0 = 1%)
        sl_pips:      Stop loss distance in pips
        symbol:       Trading symbol for pip value lookup

    Returns:
        float: Calculated lot size clamped to [0.01, 10.0]
    """
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.warning("Cannot get symbol info for %s, defaulting lot=0.01", symbol)
            return 0.01

        risk_amount = balance * (risk_percent / 100.0)

        # pip_value_per_lot in account currency
        # For most symbols: trade_tick_value / trade_tick_size * pip_size
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size

        # Determine pip size
        pip_sizes = {
            'XAUUSD': 0.10, 'XAGUSD': 0.01,
            'USDJPY': 0.01, 'EURJPY': 0.01, 'GBPJPY': 0.01
        }
        # Extract base symbol from broker variant
        base_sym = symbol[:6].upper()
        pip_size = pip_sizes.get(base_sym, 0.0001)

        pip_value_per_lot = (pip_size / tick_size) * tick_value

        if pip_value_per_lot <= 0 or sl_pips <= 0:
            return 0.01

        lot_size = risk_amount / (sl_pips * pip_value_per_lot)

        # Round to broker's lot step
        lot_step = symbol_info.volume_step
        lot_size = round(lot_size / lot_step) * lot_step

        # Clamp to broker limits
        min_lot = max(symbol_info.volume_min, 0.01)
        max_lot = min(symbol_info.volume_max, 10.0)
        lot_size = max(min_lot, min(lot_size, max_lot))

        return round(lot_size, 2)

    except Exception as e:
        logger.error("Error calculating lot size: %s", e)
        return 0.01


# ==================== API ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'timestamp': datetime.now(timezone.utc).isoformat()})


@app.route('/connect', methods=['POST'])
@require_api_key
def connect():
    """
    Verify MT5 credentials and return account information.
    Body: { telegram_id, login, password, server }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'error': 'Invalid JSON body'}), 400

    required = ('telegram_id', 'login', 'password', 'server')
    if not all(k in data for k in required):
        return jsonify({'success': False, 'error': f'Missing fields: {required}'}), 400

    telegram_id = int(data['telegram_id'])
    session = session_manager.get_or_create(telegram_id)

    with session.lock:
        success, message = _connect_mt5(
            login=int(data['login']),
            password=data['password'],
            server=data['server']
        )

        if not success:
            return jsonify({'success': False, 'error': message})

        account_info = mt5.account_info()
        result = {
            'success':    True,
            'login':      account_info.login,
            'broker':     account_info.company,
            'balance':    account_info.balance,
            'currency':   account_info.currency,
            'leverage':   account_info.leverage,
            'server':     account_info.server
        }
        session.connected = True
        mt5.shutdown()

    return jsonify(result)


@app.route('/place_order', methods=['POST'])
@require_api_key
def place_order():
    """
    Place a trade order on behalf of a user.
    Body: {
        telegram_id, login, password, server,
        symbol, direction, lot_size, entry_price,
        stop_loss, take_profit, order_type, comment
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'error': 'Invalid JSON body'}), 400

    required = (
        'telegram_id', 'login', 'password', 'server',
        'symbol', 'direction', 'lot_size', 'stop_loss',
        'take_profit', 'order_type'
    )
    if not all(k in data for k in required):
        return jsonify({'success': False, 'error': f'Missing fields: {required}'}), 400

    telegram_id = int(data['telegram_id'])
    session = session_manager.get_or_create(telegram_id)

    with session.lock:
        success, message = _connect_mt5(
            login=int(data['login']),
            password=data['password'],
            server=data['server']
        )

        if not success:
            return jsonify({'success': False, 'error': message})

        try:
            symbol = data['symbol']
            direction = data['direction'].upper()
            lot_size = float(data['lot_size'])
            stop_loss = float(data['stop_loss'])
            take_profit = float(data['take_profit'])
            order_type_str = data['order_type'].upper()
            comment = data.get('comment', 'Nix Trades')
            expiry_minutes = int(data.get('expiry_minutes', 60))

            # Ensure symbol is in Market Watch
            if not mt5.symbol_select(symbol, True):
                mt5.shutdown()
                return jsonify({'success': False, 'error': f'Symbol {symbol} not available on this broker'})

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                mt5.shutdown()
                return jsonify({'success': False, 'error': f'No tick data for {symbol}'})

            # Determine order type and price
            if direction == 'BUY':
                if order_type_str == 'MARKET':
                    mt5_order_type = mt5.ORDER_TYPE_BUY
                    price = tick.ask
                elif order_type_str == 'LIMIT':
                    mt5_order_type = mt5.ORDER_TYPE_BUY_LIMIT
                    price = float(data.get('entry_price', tick.ask))
                else:
                    mt5_order_type = mt5.ORDER_TYPE_BUY_STOP
                    price = float(data.get('entry_price', tick.ask))
            else:  # SELL
                if order_type_str == 'MARKET':
                    mt5_order_type = mt5.ORDER_TYPE_SELL
                    price = tick.bid
                elif order_type_str == 'LIMIT':
                    mt5_order_type = mt5.ORDER_TYPE_SELL_LIMIT
                    price = float(data.get('entry_price', tick.bid))
                else:
                    mt5_order_type = mt5.ORDER_TYPE_SELL_STOP
                    price = float(data.get('entry_price', tick.bid))

            symbol_info = mt5.symbol_info(symbol)
            filling_mode = symbol_info.filling_mode if symbol_info else 0

            # Select compatible filling type
            if filling_mode & mt5.ORDER_FILLING_IOC:
                filling = mt5.ORDER_FILLING_IOC
            elif filling_mode & mt5.ORDER_FILLING_FOK:
                filling = mt5.ORDER_FILLING_FOK
            else:
                filling = mt5.ORDER_FILLING_RETURN

            request_dict = {
                'action':        mt5.TRADE_ACTION_DEAL if order_type_str == 'MARKET' else mt5.TRADE_ACTION_PENDING,
                'symbol':        symbol,
                'volume':        lot_size,
                'type':          mt5_order_type,
                'price':         price,
                'sl':            stop_loss,
                'tp':            take_profit,
                'magic':         MAGIC_NUMBER,
                'comment':       comment[:31],  # MT5 comment max 31 chars
                'type_time':     mt5.ORDER_TIME_GTC if order_type_str == 'MARKET' else mt5.ORDER_TIME_SPECIFIED,
                'type_filling':  filling,
            }

            if order_type_str != 'MARKET':
                expiry_dt = datetime.now(timezone.utc) + timedelta(minutes=expiry_minutes)
                request_dict['expiration'] = int(expiry_dt.timestamp())

            result = mt5.order_send(request_dict)

            if result is None:
                error = mt5.last_error()
                mt5.shutdown()
                return jsonify({'success': False, 'error': f'order_send returned None: {error}'})

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                mt5.shutdown()
                return jsonify({
                    'success': False,
                    'error': f'Order rejected: {result.comment}',
                    'retcode': result.retcode
                })

            ticket = result.order if order_type_str != 'MARKET' else result.deal
            logger.info(
                "Order placed: user=%d %s %s %.2f lots ticket=%d",
                telegram_id, direction, symbol, lot_size, ticket
            )

        finally:
            mt5.shutdown()

    return jsonify({
        'success': True,
        'ticket':  ticket,
        'price':   price,
        'message': 'Order placed successfully'
    })


@app.route('/modify_position', methods=['POST'])
@require_api_key
def modify_position():
    """
    Modify stop loss / take profit on an open position.
    Body: { telegram_id, login, password, server, ticket, stop_loss, take_profit }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'error': 'Invalid JSON body'}), 400

    telegram_id = int(data['telegram_id'])
    session = session_manager.get_or_create(telegram_id)

    with session.lock:
        success, message = _connect_mt5(
            login=int(data['login']),
            password=data['password'],
            server=data['server']
        )

        if not success:
            return jsonify({'success': False, 'error': message})

        try:
            ticket = int(data['ticket'])
            positions = mt5.positions_get(ticket=ticket)

            if not positions:
                return jsonify({'success': False, 'error': f'Position {ticket} not found'})

            pos = positions[0]

            request_dict = {
                'action':   mt5.TRADE_ACTION_SLTP,
                'symbol':   pos.symbol,
                'position': ticket,
                'sl':       float(data.get('stop_loss', pos.sl)),
                'tp':       float(data.get('take_profit', pos.tp)),
                'magic':    MAGIC_NUMBER
            }

            result = mt5.order_send(request_dict)

            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error = mt5.last_error() if result is None else result.comment
                return jsonify({'success': False, 'error': f'Modify failed: {error}'})

        finally:
            mt5.shutdown()

    return jsonify({'success': True, 'message': 'Position modified'})


@app.route('/close_partial', methods=['POST'])
@require_api_key
def close_partial():
    """
    Close a percentage of an open position (e.g., 50% at TP1).
    Body: { telegram_id, login, password, server, ticket, close_percent }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'error': 'Invalid JSON body'}), 400

    telegram_id = int(data['telegram_id'])
    close_percent = float(data.get('close_percent', 50.0))

    session = session_manager.get_or_create(telegram_id)

    with session.lock:
        success, message = _connect_mt5(
            login=int(data['login']),
            password=data['password'],
            server=data['server']
        )

        if not success:
            return jsonify({'success': False, 'error': message})

        try:
            ticket = int(data['ticket'])
            positions = mt5.positions_get(ticket=ticket)

            if not positions:
                return jsonify({'success': False, 'error': f'Position {ticket} not found'})

            pos = positions[0]
            symbol_info = mt5.symbol_info(pos.symbol)

            close_volume = round(pos.volume * (close_percent / 100.0), 2)
            close_volume = max(symbol_info.volume_min, close_volume)

            tick = mt5.symbol_info_tick(pos.symbol)
            close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

            filling_mode = symbol_info.filling_mode if symbol_info else 0
            if filling_mode & mt5.ORDER_FILLING_IOC:
                filling = mt5.ORDER_FILLING_IOC
            elif filling_mode & mt5.ORDER_FILLING_FOK:
                filling = mt5.ORDER_FILLING_FOK
            else:
                filling = mt5.ORDER_FILLING_RETURN

            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

            request_dict = {
                'action':       mt5.TRADE_ACTION_DEAL,
                'symbol':       pos.symbol,
                'volume':       close_volume,
                'type':         close_type,
                'position':     ticket,
                'price':        close_price,
                'magic':        MAGIC_NUMBER,
                'comment':      'NixTrades TP1',
                'type_filling': filling
            }

            result = mt5.order_send(request_dict)

            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error = mt5.last_error() if result is None else result.comment
                return jsonify({'success': False, 'error': f'Partial close failed: {error}'})

            logger.info(
                "Partial close: user=%d ticket=%d closed %.2f lots",
                telegram_id, ticket, close_volume
            )

        finally:
            mt5.shutdown()

    return jsonify({
        'success':       True,
        'closed_volume': close_volume,
        'close_price':   close_price,
        'message':       f'Closed {close_percent:.0f}% of position'
    })


@app.route('/get_positions', methods=['POST'])
@require_api_key
def get_positions():
    """
    Retrieve all open positions for a user's account.
    Body: { telegram_id, login, password, server }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'error': 'Invalid JSON body'}), 400

    telegram_id = int(data['telegram_id'])
    session = session_manager.get_or_create(telegram_id)

    with session.lock:
        success, message = _connect_mt5(
            login=int(data['login']),
            password=data['password'],
            server=data['server']
        )

        if not success:
            return jsonify({'success': False, 'error': message})

        try:
            positions_raw = mt5.positions_get()
            if positions_raw is None:
                positions_raw = []

            positions = []
            for pos in positions_raw:
                positions.append({
                    'ticket':        pos.ticket,
                    'symbol':        pos.symbol,
                    'type':          'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'volume':        pos.volume,
                    'price_open':    pos.price_open,
                    'price_current': pos.price_current,
                    'sl':            pos.sl,
                    'tp':            pos.tp,
                    'profit':        pos.profit,
                    'magic':         pos.magic,
                    'comment':       pos.comment
                })

        finally:
            mt5.shutdown()

    return jsonify({'success': True, 'positions': positions})


@app.route('/calculate_lot', methods=['POST'])
@require_api_key
def calculate_lot():
    """
    Calculate lot size for a given risk configuration.
    Body: { telegram_id, login, password, server, symbol, risk_percent, sl_pips }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'error': 'Invalid JSON body'}), 400

    telegram_id = int(data['telegram_id'])
    session = session_manager.get_or_create(telegram_id)

    with session.lock:
        success, message = _connect_mt5(
            login=int(data['login']),
            password=data['password'],
            server=data['server']
        )

        if not success:
            return jsonify({'success': False, 'error': message})

        try:
            account_info = mt5.account_info()
            balance = account_info.balance
            risk_percent = float(data.get('risk_percent', 1.0))
            sl_pips = float(data['sl_pips'])
            symbol = data['symbol']

            lot_size = _calculate_lot_size(balance, risk_percent, sl_pips, symbol)

        finally:
            mt5.shutdown()

    return jsonify({
        'success':    True,
        'lot_size':   lot_size,
        'balance':    balance,
        'risk_amount': round(balance * risk_percent / 100.0, 2)
    })


# ==================== BACKGROUND CLEANUP ====================

def _run_cleanup():
    """Periodically remove expired sessions."""
    while True:
        time.sleep(60)
        session_manager.cleanup_expired()


cleanup_thread = threading.Thread(target=_run_cleanup, daemon=True)
cleanup_thread.start()

# ==================== ENTRY POINT ====================

if __name__ == '__main__':
    logger.info(
        "MT5 Worker Service starting on %s:%d", WORKER_HOST, WORKER_PORT
    )
    logger.info("Max concurrent sessions: %d", MAX_SESSIONS)
    logger.info(
        "Session idle timeout: %d seconds", SESSION_TIMEOUT_SECONDS
    )

    app.run(
        host=WORKER_HOST,
        port=WORKER_PORT,
        debug=False,
        threaded=True
    )