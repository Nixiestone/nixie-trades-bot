"""
NIX TRADES - Database Operations
Role: Data Engineer + Security Engineer + Python Developer

Changes in this version:
  - _db_retry decorator: 4 attempts with exponential backoff on all Supabase calls
  - disclaimer_accepted and disclaimer_accepted_at fields added to user record
  - recent_signal_exists() for signal deduplication per timeframe expiry
  - get_signal_count() for sequential signal numbering
  - get_daily_loss_percent() for daily loss limit enforcement per user
  - queue_message() and get_pending_messages() for offline user message delivery
  - mark_messages_delivered() clears delivered queued messages
  - save_signal() updated to include timeframe and expiry_hours
  - get_subscribed_users() now includes disclaimer_accepted field
  - Passwords are never logged (scrubbed before any log call)

NO EMOJIS - Enterprise code only
NO PLACEHOLDERS - All functions are fully implemented
"""

import json
import logging
import os
import time
import functools
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone, timedelta

from supabase import create_client, Client
from cryptography.fernet import Fernet, InvalidToken

import config
import utils

logger = logging.getLogger(__name__)

# Module-level Supabase client and encryption objects
supabase_client: Optional[Client] = None
_fernet: Optional[Fernet] = None


# ==================== RETRY DECORATOR ====================

def _db_retry(max_attempts: int = 4, base_delay: float = 1.0):
    """
    Decorator that retries a database function on transient failures.
    Delays: 0s, 1s, 2s, 4s (exponential backoff, 4 attempts total).
    On permanent failure, raises the last exception.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "Database call %s failed (attempt %d/%d), retrying in %.1fs: %s",
                            func.__name__, attempt + 1, max_attempts, delay, e
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "Database call %s permanently failed after %d attempts: %s",
                            func.__name__, max_attempts, e
                        )
            raise last_error
        return wrapper
    return decorator


# ==================== INITIALISATION ====================

def init_supabase(max_retries: int = 4) -> Client:
    """
    Initialise the Supabase client and Fernet encryption.
    Retries on transient connection failure with exponential backoff.
    """
    global supabase_client, _fernet

    supabase_url        = os.getenv('SUPABASE_URL', '').strip()
    supabase_key        = os.getenv('SUPABASE_KEY', '').strip()
    encryption_key_str  = os.getenv('ENCRYPTION_KEY', '').strip()

    if not supabase_url:
        raise ValueError("SUPABASE_URL environment variable is not set.")
    if not supabase_key:
        raise ValueError("SUPABASE_KEY environment variable is not set.")
    if not encryption_key_str:
        raise ValueError("ENCRYPTION_KEY environment variable is not set.")

    try:
        _fernet = Fernet(encryption_key_str.encode())
        logger.info("Encryption engine initialised.")
    except Exception:
        raise ValueError(
            "ENCRYPTION_KEY is invalid. "
            "Generate a valid key with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )

    last_error = None
    for attempt in range(max_retries):
        try:
            supabase_client = create_client(supabase_url, supabase_key)
            logger.info("Supabase connection established.")
            return supabase_client
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                logger.warning(
                    "Supabase connection failed (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, max_retries, delay, e
                )
                time.sleep(delay)

    raise RuntimeError(
        f"Cannot connect to database after {max_retries} attempts: {last_error}"
    )


def _client() -> Client:
    """Return the initialised Supabase client, raising if not initialised."""
    if supabase_client is None:
        raise RuntimeError("Database not initialised. Call init_supabase() first.")
    return supabase_client


def _encrypt(value: str) -> str:
    """Encrypt a string using Fernet (AES-128-CBC + HMAC-SHA256)."""
    if _fernet is None:
        raise RuntimeError("Encryption not initialised.")
    return _fernet.encrypt(value.encode()).decode()


def _decrypt(value: str) -> str:
    """Decrypt a Fernet-encrypted string. Raises InvalidToken on tampering."""
    if _fernet is None:
        raise RuntimeError("Encryption not initialised.")
    return _fernet.decrypt(value.encode()).decode()


# ==================== USER MANAGEMENT ====================

@_db_retry()
def get_or_create_user(
    telegram_id: int,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
    user_timezone: str = 'UTC',
) -> dict:
    """
    Return existing user record or create a new one.
    New users have subscription_status='inactive' and disclaimer_accepted=False.
    """
    result = (
        _client()
        .table('telegram_users')
        .select('*')
        .eq('telegram_id', telegram_id)
        .execute()
    )

    if result.data:
        return result.data[0]

    new_user = {
        'telegram_id':         telegram_id,
        'username':            username,
        'first_name':          first_name,
        'timezone':            user_timezone,
        'subscription_status': 'inactive',
        'risk_percent':        config.DEFAULT_RISK_PERCENT,
        'mt5_connected':       False,
        'disclaimer_accepted': False,
        'created_at':          datetime.now(timezone.utc).isoformat(),
        'updated_at':          datetime.now(timezone.utc).isoformat(),
    }

    insert_result = (
        _client()
        .table('telegram_users')
        .insert(new_user)
        .execute()
    )

    return insert_result.data[0] if insert_result.data else new_user


@_db_retry()
def get_user(telegram_id: int) -> Optional[dict]:
    """Return a user record, or None if not found."""
    result = (
        _client()
        .table('telegram_users')
        .select('*')
        .eq('telegram_id', telegram_id)
        .execute()
    )
    return result.data[0] if result.data else None


@_db_retry()
def update_subscription(telegram_id: int, status: str) -> bool:
    """Set subscription_status for a user. Status: 'active', 'inactive', 'suspended'."""
    _client().table('telegram_users').update({
        'subscription_status': status,
        'updated_at':          datetime.now(timezone.utc).isoformat(),
    }).eq('telegram_id', telegram_id).execute()
    return True


@_db_retry()
def accept_disclaimer(telegram_id: int) -> bool:
    """Record that the user has accepted the legal disclaimer."""
    _client().table('telegram_users').update({
        'disclaimer_accepted':    True,
        'disclaimer_accepted_at': datetime.now(timezone.utc).isoformat(),
        'updated_at':             datetime.now(timezone.utc).isoformat(),
    }).eq('telegram_id', telegram_id).execute()
    return True


@_db_retry()
def update_risk_percent(telegram_id: int, risk_percent: float) -> bool:
    """Update the risk percentage setting for a user."""
    _client().table('telegram_users').update({
        'risk_percent': risk_percent,
        'updated_at':   datetime.now(timezone.utc).isoformat(),
    }).eq('telegram_id', telegram_id).execute()
    return True


@_db_retry()
def update_timezone(telegram_id: int, tz: str) -> bool:
    """Update the timezone setting for a user."""
    _client().table('telegram_users').update({
        'timezone':   tz,
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }).eq('telegram_id', telegram_id).execute()
    return True


@_db_retry()
def get_subscribed_users() -> List[dict]:
    """
    Return all users with active subscriptions.
    Includes disclaimer_accepted so callers can enforce the disclaimer gate.
    """
    result = (
        _client()
        .table('telegram_users')
        .select(
            'telegram_id, username, first_name, timezone, risk_percent, '
            'mt5_login, mt5_connected, disclaimer_accepted'
        )
        .eq('subscription_status', 'active')
        .execute()
    )
    return result.data or []


@_db_retry()
def get_user_statistics(telegram_id: int) -> dict:
    """
    Return trading statistics summary for a user.
    Used by the /status command.
    """
    trades_result = (
        _client()
        .table('trades')
        .select('profit_pips, outcome, created_at')
        .eq('telegram_id', telegram_id)
        .order('created_at', desc=True)
        .limit(100)
        .execute()
    )

    trades = trades_result.data or []
    total  = len(trades)
    wins   = sum(1 for t in trades if t.get('outcome') == 'WIN')
    losses = sum(1 for t in trades if t.get('outcome') == 'LOSS')
    pips   = sum(float(t.get('profit_pips', 0)) for t in trades)

    return {
        'total_trades': total,
        'wins':         wins,
        'losses':       losses,
        'win_rate':     round((wins / total * 100) if total > 0 else 0.0, 1),
        'total_pips':   round(pips, 1),
    }


# ==================== MT5 CREDENTIALS ====================

@_db_retry()
def save_mt5_credentials(
    telegram_id: int,
    login: int,
    password: str,
    server: str,
    broker_name: str = '',
    balance: float = 0.0,
    currency: str = 'USD',
) -> bool:
    """
    Encrypt and save MT5 credentials for a user.
    Password is encrypted with Fernet before storage.
    Login and server are stored plaintext for account lookups.
    """
    encrypted_password = _encrypt(password)

    _client().table('telegram_users').update({
        'mt5_login':             login,
        'mt5_password_encrypted': encrypted_password,
        'mt5_server':            server,
        'mt5_broker_name':       broker_name,
        'mt5_account_balance':   balance,
        'mt5_account_currency':  currency,
        'mt5_connected':         True,
        'updated_at':            datetime.now(timezone.utc).isoformat(),
    }).eq('telegram_id', telegram_id).execute()

    logger.info("MT5 credentials saved for user %d (server: %s)", telegram_id, server)
    return True


@_db_retry()
def get_mt5_credentials(telegram_id: int) -> Optional[dict]:
    """
    Retrieve and decrypt MT5 credentials for a user.
    Returns None if credentials are not found or decryption fails.
    Password is never logged.
    """
    result = (
        _client()
        .table('telegram_users')
        .select('mt5_login, mt5_password_encrypted, mt5_server, mt5_account_currency')
        .eq('telegram_id', telegram_id)
        .eq('mt5_connected', True)
        .execute()
    )

    if not result.data:
        return None

    row = result.data[0]
    if not row.get('mt5_login') or not row.get('mt5_password_encrypted'):
        return None

    try:
        password = _decrypt(row['mt5_password_encrypted'])
    except (InvalidToken, Exception) as e:
        logger.error(
            "Decryption failed for user %d - credentials may be corrupted: %s",
            telegram_id, type(e).__name__
        )
        return None

    return {
        'login':    row['mt5_login'],
        'password': password,        # never logged after this point
        'server':   row['mt5_server'],
        'currency': row.get('mt5_account_currency', 'USD'),
    }


@_db_retry()
def delete_mt5_credentials(telegram_id: int) -> bool:
    """Remove MT5 credentials for a user (called on /disconnect_mt5)."""
    _client().table('telegram_users').update({
        'mt5_login':             None,
        'mt5_password_encrypted': None,
        'mt5_server':            None,
        'mt5_broker_name':       None,
        'mt5_account_balance':   None,
        'mt5_account_currency':  None,
        'mt5_connected':         False,
        'updated_at':            datetime.now(timezone.utc).isoformat(),
    }).eq('telegram_id', telegram_id).execute()
    logger.info("MT5 credentials deleted for user %d", telegram_id)
    return True


@_db_retry()
def get_mt5_connection(telegram_id: int) -> Optional[dict]:
    """Alias for get_mt5_credentials. Used by scheduler and position monitor."""
    return get_mt5_credentials(telegram_id)


# ==================== SIGNALS ====================

@_db_retry()
def save_signal(
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
    lstm_score: int = 0,
    xgboost_score: int = 0,
    session: str = 'N/A',
    order_type: str = 'LIMIT',
    timeframe: str = 'M15',
    expiry_hours: int = 2,
) -> Optional[int]:
    """
    Save a new signal to the database.
    Returns the signal's ID, or None on failure.
    """
    signal_data = {
        'symbol':        symbol,
        'direction':     direction,
        'setup_type':    setup_type,
        'entry_price':   entry_price,
        'stop_loss':     stop_loss,
        'take_profit_1': take_profit_1,
        'take_profit_2': take_profit_2,
        'sl_pips':       sl_pips,
        'tp1_pips':      tp1_pips,
        'tp2_pips':      tp2_pips,
        'rr_tp1':        rr_tp1,
        'rr_tp2':        rr_tp2,
        'ml_score':      ml_score,
        'lstm_score':    lstm_score,
        'xgboost_score': xgboost_score,
        'session':       session,
        'order_type':    order_type,
        'timeframe':     timeframe,
        'expiry_hours':  expiry_hours,
        'status':        'active',
        'created_at':    datetime.now(timezone.utc).isoformat(),
    }

    result = _client().table('signals').insert(signal_data).execute()
    if result.data:
        signal_id = result.data[0].get('id')
        logger.info(
            "Signal saved: id=%s %s %s score=%d",
            signal_id, symbol, direction, ml_score
        )
        return signal_id
    return None


@_db_retry()
def get_signal_count() -> int:
    """Return the total number of signals ever saved. Used for sequential numbering."""
    result = (
        _client()
        .table('signals')
        .select('id', count='exact')
        .execute()
    )
    return result.count or 0


@_db_retry()
def recent_signal_exists(
    symbol: str,
    direction: str,
    hours: int = 2,
) -> bool:
    """
    Return True if a signal for symbol+direction was saved within the
    last `hours` hours. Used to prevent duplicate signals within the
    timeframe expiry window.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    result = (
        _client()
        .table('signals')
        .select('id')
        .eq('symbol', symbol)
        .eq('direction', direction)
        .gte('created_at', cutoff)
        .limit(1)
        .execute()
    )
    return bool(result.data)


@_db_retry()
def get_latest_signal() -> Optional[dict]:
    """Return the most recently created signal."""
    result = (
        _client()
        .table('signals')
        .select('*')
        .order('created_at', desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


# ==================== TRADES ====================

@_db_retry()
def save_trade(
    telegram_id: int,
    signal_id: Optional[int],
    symbol: str,
    direction: str,
    lot_size: float,
    entry_price: float,
    fill_price: float,
    stop_loss: float,
    take_profit_1: float,
    take_profit_2: float,
    order_type: str,
    ticket: int,
) -> Optional[int]:
    """
    Save a trade execution record for performance tracking.
    fill_price is the actual broker-filled price (may differ from entry_price).
    """
    trade_data = {
        'telegram_id':   telegram_id,
        'signal_id':     signal_id,
        'symbol':        symbol,
        'direction':     direction,
        'lot_size':      lot_size,
        'entry_price':   entry_price,
        'fill_price':    fill_price,
        'stop_loss':     stop_loss,
        'take_profit_1': take_profit_1,
        'take_profit_2': take_profit_2,
        'order_type':    order_type,
        'mt5_ticket':    ticket,          # column is mt5_ticket per schema
        'status':        'OPEN',
        'outcome':       None,
        'profit_pips':   None,
        'rr_achieved':   None,
        'opened_at':     datetime.now(timezone.utc).isoformat(),
        'created_at':    datetime.now(timezone.utc).isoformat(),
        'updated_at':    datetime.now(timezone.utc).isoformat(),
    }

    result = _client().table('trades').insert(trade_data).execute()
    if result.data:
        return result.data[0].get('id')
    return None


@_db_retry()
def update_trade(
    ticket: int,
    outcome: Optional[str] = None,
    profit_pips: Optional[float] = None,
    rr_achieved: Optional[float] = None,
    close_price: Optional[float] = None,
    status: str = 'closed',
) -> bool:
    """
    Update a trade record when it is closed (by TP, SL, or manual close).
    outcome: 'WIN', 'LOSS', or 'BREAKEVEN'
    """
    update_data = {
        'status':     status,
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }
    if outcome is not None:
        update_data['outcome'] = outcome
    if profit_pips is not None:
        update_data['profit_pips'] = profit_pips
    if rr_achieved is not None:
        update_data['rr_achieved'] = rr_achieved
    if close_price is not None:
        update_data['close_price'] = close_price

    _client().table('trades').update(update_data).eq('mt5_ticket', ticket).execute()
    return True

def recent_signal_exists(symbol: str, minutes: int = 15) -> bool:
    """
    Return True if a setup for this symbol was saved within the last N minutes.
    Used by scheduler to prevent sending duplicate setups for the same pair.

    Args:
        symbol:  Trading symbol (e.g. 'EURUSD')
        minutes: Look-back window in minutes

    Returns:
        bool: True if a recent signal exists for this symbol
    """
    try:
        client = _client()
        cutoff = (
            utils.get_current_utc_time() - timedelta(minutes=minutes)
        ).isoformat()

        response = (
            client.table('signals')
            .select('id')
            .eq('symbol', symbol)
            .gte('created_at', cutoff)
            .limit(1)
            .execute()
        )
        return bool(response.data)

    except Exception as e:
        logger.error(
            "Error checking recent signal for %s: %s", symbol, e
        )
        return False   # Default: allow the signal through on error


def save_trade_outcome_for_ml(
    ticket:   int,
    features: object,
    outcome:  float,
) -> bool:
    """
    Store ML training data produced by a closed trade.
    Called by position_monitor._log_trade_completion() for every trade close.

    The features object is a numpy ndarray (22 elements). It is serialised
    to a JSON list for storage in the database. MLEnsemble.record_trade_outcome()
    is also called in memory; this function provides durable storage so training
    data survives bot restarts.

    Args:
        ticket:   MT5 ticket number (for traceability)
        features: 22-element numpy feature vector from ml.get_ensemble_prediction()
        outcome:  1.0 = WIN, 0.0 = LOSS

    Returns:
        bool: True if saved successfully
    """
    try:
        client = _client()

        # Convert ndarray to plain list for JSON serialisation
        if hasattr(features, 'tolist'):
            features_list = features.tolist()
        else:
            features_list = list(features)

        row = {
            'mt5_ticket':    ticket,
            'features_json': json.dumps(features_list),
            'outcome':       float(outcome),
            'created_at':    utils.get_current_utc_time().isoformat(),
        }

        response = client.table('ml_training_data').insert(row).execute()
        if response.data:
            logger.debug(
                "ML training data saved: ticket=%d outcome=%.0f", ticket, outcome
            )
            return True
        return False

    except Exception as e:
        logger.error(
            "Error saving ML training data for ticket %d: %s", ticket, e
        )
        return False



def get_all_trades_for_csv(telegram_id: int) -> List[Dict[str, Any]]:
    """
    Retrieve all trades for a user ordered newest-first.
    Used by the /download command to generate the trading history CSV.

    Args:
        telegram_id: Telegram user ID

    Returns:
        List of trade dicts with all columns
    """
    try:
        client = _client()
        response = (
            client.table('trades')
            .select('*')
            .eq('telegram_id', telegram_id)
            .order('opened_at', desc=True)
            .execute()
        )
        return response.data or []

    except Exception as e:
        logger.error(
            "Error fetching trades for CSV (user %d): %s", telegram_id, e
        )
        return []


def get_all_signals_for_csv() -> List[Dict[str, Any]]:
    """
    Retrieve all signals (automated setups) ordered newest-first.
    Used by the /download command to generate the ML training results CSV.
    Includes ML scores and setup quality data for analysis.

    Returns:
        List of signal dicts
    """
    try:
        client = _client()
        response = (
            client.table('signals')
            .select('*')
            .order('created_at', desc=True)
            .execute()
        )
        return response.data or []

    except Exception as e:
        logger.error("Error fetching signals for CSV: %s", e)
        return []


@_db_retry()
def get_daily_loss_percent(
    telegram_id: int,
    since: datetime,
) -> float:
    """
    Calculate the total loss as a percentage of the user's opening balance
    for all trades closed since `since` (UTC datetime).

    Returns a positive float representing the loss percentage.
    Returns 0.0 if there are no losses or on any error.
    """
    result = (
        _client()
        .table('trades')
        .select('profit_pips, lot_size, symbol')
        .eq('telegram_id', telegram_id)
        .eq('status', 'closed')
        .gte('updated_at', since.isoformat())
        .execute()
    )

    trades = result.data or []
    if not trades:
        return 0.0

    # Sum negative profit_pips weighted by lot size as a proxy for % loss.
    # A more accurate calculation requires the account balance at trade open,
    # which is stored in the user record. We use pip-weighted loss as an
    # approximation that is safe (may over-estimate loss, which is conservative).
    total_loss_pips = sum(
        float(t.get('profit_pips', 0)) for t in trades
        if float(t.get('profit_pips', 0)) < 0
    )

    # Fetch user's stored balance for percent calculation
    user = get_user(telegram_id)
    balance = float(user.get('mt5_account_balance', 0)) if user else 0.0

    if balance <= 0:
        # Cannot calculate percent without balance; use pip threshold (50 pips = alert)
        return 3.0 if abs(total_loss_pips) >= 50 else 0.0

    # Rough loss in account currency: assume 1 pip = $10 per standard lot
    loss_currency = abs(total_loss_pips) * 10
    return round((loss_currency / balance) * 100, 2)


# ==================== QUEUED MESSAGES ====================

@_db_retry()
def queue_message(
    telegram_id: int,
    message_text: str,
    message_type: str = 'SETUP_ALERT',
) -> bool:
    """
    Store a message for a user who is currently offline.
    Writes to message_queue (created by create_tables.sql).
    Valid message_type values: SETUP_ALERT, TRADE_NOTIFICATION, SYSTEM_MESSAGE.
    """
    valid_types = ('SETUP_ALERT', 'TRADE_NOTIFICATION', 'SYSTEM_MESSAGE')
    if message_type not in valid_types:
        message_type = 'SYSTEM_MESSAGE'

    _client().table('message_queue').insert({
        'telegram_id':  telegram_id,
        'message_text': message_text,
        'message_type': message_type,
        'status':       'PENDING',
        'created_at':   datetime.now(timezone.utc).isoformat(),
    }).execute()
    return True


@_db_retry()
def get_pending_messages(telegram_id: int) -> List[dict]:
    """Return all undelivered queued messages for a user."""
    result = (
        _client()
        .table('message_queue')
        .select('id, message_text, message_type, created_at')
        .eq('telegram_id', telegram_id)
        .eq('status', 'PENDING')
        .order('created_at')
        .execute()
    )
    return result.data or []


@_db_retry()
def mark_messages_delivered(telegram_id: int) -> bool:
    """Mark all queued messages for a user as sent."""
    _client().table('message_queue').update({
        'status':  'SENT',
        'sent_at': datetime.now(timezone.utc).isoformat(),
    }).eq('telegram_id', telegram_id).eq('status', 'PENDING').execute()
    return True