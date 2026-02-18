"""
NIX TRADES - Database Operations
Supabase PostgreSQL integration with encrypted credential storage
Role: Python Developer + Data Engineer + Security Engineer
Fixes:
  - Added missing 'from datetime import timedelta' (used in get_latest_signal, get_todays_news, clear_old_news)
  - Added exponential backoff retry on Supabase init (Data Engineer)
  - Added proper Fernet key validation before use (Security Engineer)
  - MT5 credentials now use column-level encryption for password only;
    login and server are stored plaintext to allow broker account lookup
  - Added get_subscribed_users() function referenced by scheduler but missing from this file
  - Added get_user_statistics() function referenced by bot.py but missing from this file
  - Added save_trade() and update_trade() functions referenced by position_monitor
  - Fixed timezone parameter name collision with stdlib in create_user and update_timezone
NO EMOJIS - Professional code only
"""

import logging
import os
import time
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


# ==================== INITIALISATION ====================

def init_supabase(max_retries: int = 3) -> Client:
    """
    Initialise Supabase client with exponential backoff retry.

    Args:
        max_retries: Number of connection attempts before raising

    Returns:
        Client: Supabase client instance

    Raises:
        ValueError: If required environment variables are missing
        RuntimeError: If connection cannot be established after retries
    """
    global supabase_client, _fernet

    supabase_url = os.getenv('SUPABASE_URL', '').strip()
    supabase_key = os.getenv('SUPABASE_KEY', '').strip()
    encryption_key_str = os.getenv('ENCRYPTION_KEY', '').strip()

    if not supabase_url:
        raise ValueError(
            "Database configuration is missing. Check your environment setup."
        )
    if not supabase_key:
        raise ValueError(
            "Database authentication is missing. Check your environment setup."
        )
    if not encryption_key_str:
        raise ValueError(
            "Security configuration is missing. Check your environment setup."
        )

    # Validate encryption key format before use
    try:
        _fernet = Fernet(encryption_key_str.encode())
        logger.info("Encryption initialised successfully.")
    except Exception as e:
        raise ValueError(
            "Security configuration is invalid. Please check your setup."
        )

    last_error = None
    for attempt in range(max_retries):
        try:
            supabase_client = create_client(supabase_url, supabase_key)

            # Verify connectivity with a lightweight query
            supabase_client.table('telegram_users').select('telegram_id').limit(1).execute()
            logger.info("Supabase client initialised and connection verified.")
            return supabase_client

        except Exception as e:
            last_error = e
            sleep_time = 2 ** attempt
            logger.warning(
                "Supabase connection attempt %d/%d failed: %s. Retrying in %ds.",
                attempt + 1, max_retries, e, sleep_time
            )
            if attempt < max_retries - 1:
                time.sleep(sleep_time)

    raise RuntimeError(
        "Database connection could not be established. Please try again later or contact support."
    )


def get_client() -> Client:
    """
    Return the Supabase client, initialising if necessary.

    Returns:
        Client: Supabase client instance
    """
    global supabase_client
    if supabase_client is None:
        supabase_client = init_supabase()
    return supabase_client


def _get_fernet() -> Fernet:
    """
    Return the Fernet instance, raising if not initialised.

    Returns:
        Fernet: Encryption object
    """
    global _fernet
    if _fernet is None:
        raise RuntimeError("Encryption not initialised. Call init_supabase() first.")
    return _fernet


# ==================== ENCRYPTION HELPERS ====================

def encrypt_value(plaintext: str) -> str:
    """
    Encrypt a plaintext string using Fernet symmetric encryption.

    Args:
        plaintext: String to encrypt

    Returns:
        str: Base64-encoded ciphertext
    """
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt_value(ciphertext: str) -> str:
    """
    Decrypt a Fernet-encrypted string.

    Args:
        ciphertext: Base64-encoded ciphertext

    Returns:
        str: Decrypted plaintext

    Raises:
        InvalidToken: If the ciphertext is corrupt or the key has changed
    """
    return _get_fernet().decrypt(ciphertext.encode()).decode()


# ==================== USER MANAGEMENT ====================

def create_user(
    telegram_id: int,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
    user_timezone: str = 'UTC'
) -> Optional[Dict[str, Any]]:
    """
    Create a new user record in the database.

    Args:
        telegram_id:    Telegram user ID
        username:       Telegram username (without @)
        first_name:     User's first name
        user_timezone:  IANA timezone string (renamed from 'timezone' to avoid stdlib collision)

    Returns:
        dict: Created user record, or None if failed
    """
    try:
        client = get_client()
        now = utils.get_current_utc_time().isoformat()

        user_data = {
            'telegram_id':         telegram_id,
            'username':            username,
            'first_name':          first_name,
            'timezone':            user_timezone,
            'subscription_status': 'inactive',
            'risk_percent':        config.DEFAULT_RISK_PERCENT,
            'created_at':          now,
            'updated_at':          now
        }

        response = client.table('telegram_users').insert(user_data).execute()

        if response.data:
            logger.info("Created user telegram_id=%d", telegram_id)
            return response.data[0]

        logger.error("Failed to create user telegram_id=%d: empty response", telegram_id)
        return None

    except Exception as e:
        logger.error("Error creating user %d: %s", telegram_id, e)
        return None


def get_user(telegram_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve a user record by Telegram ID.

    Args:
        telegram_id: Telegram user ID

    Returns:
        dict: User record, or None if not found
    """
    try:
        client = get_client()
        response = (
            client.table('telegram_users')
            .select('*')
            .eq('telegram_id', telegram_id)
            .execute()
        )
        return response.data[0] if response.data else None

    except Exception as e:
        logger.error("Error fetching user %d: %s", telegram_id, e)
        return None


def get_or_create_user(
    telegram_id: int,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
    user_timezone: str = 'UTC'
) -> Optional[Dict[str, Any]]:
    """
    Retrieve user or create if not exists.

    Args:
        telegram_id:    Telegram user ID
        username:       Telegram username
        first_name:     User's first name
        user_timezone:  IANA timezone string

    Returns:
        dict: User record
    """
    user = get_user(telegram_id)
    if user is None:
        user = create_user(telegram_id, username, first_name, user_timezone)
    return user


def update_user(telegram_id: int, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Update user fields.

    Args:
        telegram_id: Telegram user ID
        updates:     Dictionary of fields to update

    Returns:
        dict: Updated user record, or None if failed
    """
    try:
        client = get_client()
        updates['updated_at'] = utils.get_current_utc_time().isoformat()

        response = (
            client.table('telegram_users')
            .update(updates)
            .eq('telegram_id', telegram_id)
            .execute()
        )

        if response.data:
            logger.debug("Updated user %d: %s", telegram_id, list(updates.keys()))
            return response.data[0]

        logger.error("Failed to update user %d", telegram_id)
        return None

    except Exception as e:
        logger.error("Error updating user %d: %s", telegram_id, e)
        return None


def activate_subscription(telegram_id: int) -> bool:
    """Activate user subscription."""
    try:
        updates = {
            'subscription_status': 'active',
            'trial_started_at':    utils.get_current_utc_time().isoformat()
        }
        return update_user(telegram_id, updates) is not None
    except Exception as e:
        logger.error("Error activating subscription for %d: %s", telegram_id, e)
        return False


def deactivate_subscription(telegram_id: int) -> bool:
    """Deactivate user subscription."""
    try:
        return update_user(telegram_id, {'subscription_status': 'inactive'}) is not None
    except Exception as e:
        logger.error("Error deactivating subscription for %d: %s", telegram_id, e)
        return False


def update_timezone(telegram_id: int, new_timezone: str) -> bool:
    """
    Update user timezone.

    Args:
        telegram_id:  Telegram user ID
        new_timezone: IANA timezone string

    Returns:
        bool: True if successful
    """
    try:
        return update_user(telegram_id, {'timezone': new_timezone}) is not None
    except Exception as e:
        logger.error("Error updating timezone for %d: %s", telegram_id, e)
        return False


def update_risk_percent(telegram_id: int, risk_percent: float) -> bool:
    """
    Update user risk percentage.

    Args:
        telegram_id:  Telegram user ID
        risk_percent: Risk per trade (0.1 - 5.0)

    Returns:
        bool: True if successful
    """
    try:
        if not utils.validate_risk_percent(risk_percent):
            logger.warning(
                "Invalid risk_percent %.2f for user %d", risk_percent, telegram_id
            )
            return False
        return update_user(telegram_id, {'risk_percent': risk_percent}) is not None
    except Exception as e:
        logger.error("Error updating risk_percent for %d: %s", telegram_id, e)
        return False


def get_subscribed_users() -> List[Dict[str, Any]]:
    """
    Retrieve all users with active subscriptions.
    Used by the scheduler to send daily alerts and setup notifications.

    Returns:
        list: List of user dicts
    """
    try:
        client = get_client()
        response = (
            client.table('telegram_users')
            .select('*')
            .eq('subscription_status', 'active')
            .execute()
        )
        return response.data if response.data else []

    except Exception as e:
        logger.error("Error fetching subscribed users: %s", e)
        return []


def get_users_with_mt5() -> List[Dict[str, Any]]:
    """
    Retrieve all subscribed users who have MT5 connected.
    Used by the position monitor and trade executor.

    Returns:
        list: List of user dicts with mt5_connected=True
    """
    try:
        client = get_client()
        response = (
            client.table('telegram_users')
            .select('*')
            .eq('subscription_status', 'active')
            .eq('mt5_connected', True)
            .execute()
        )
        return response.data if response.data else []

    except Exception as e:
        logger.error("Error fetching MT5-connected users: %s", e)
        return []


def get_user_statistics(telegram_id: int) -> Optional[Dict[str, Any]]:
    """
    Calculate trading statistics for a user from their trade history.

    Args:
        telegram_id: Telegram user ID

    Returns:
        dict with keys: total_setups, successful_trades, win_rate, total_profit, avg_rr
        or None if no data
    """
    try:
        client = get_client()
        response = (
            client.table('trades')
            .select('realized_pnl, rr_achieved, status')
            .eq('telegram_id', telegram_id)
            .eq('status', 'CLOSED')
            .execute()
        )

        trades = response.data if response.data else []

        if not trades:
            return {
                'total_setups':     0,
                'successful_trades': 0,
                'win_rate':         0.0,
                'total_profit':     0.0,
                'avg_rr':           0.0
            }

        total = len(trades)
        wins = sum(1 for t in trades if (t.get('realized_pnl') or 0) > 0)
        total_profit = sum((t.get('realized_pnl') or 0) for t in trades)
        avg_rr_values = [t['rr_achieved'] for t in trades if t.get('rr_achieved') is not None]
        avg_rr = sum(avg_rr_values) / len(avg_rr_values) if avg_rr_values else 0.0

        return {
            'total_setups':      total,
            'successful_trades': wins,
            'win_rate':          round((wins / total) * 100, 1) if total > 0 else 0.0,
            'total_profit':      round(total_profit, 2),
            'avg_rr':            round(avg_rr, 2)
        }

    except Exception as e:
        logger.error("Error fetching statistics for user %d: %s", telegram_id, e)
        return None


# ==================== MT5 CREDENTIAL MANAGEMENT ====================

def save_mt5_credentials(
    telegram_id: int,
    login: int,
    password: str,
    server: str,
    broker_name: str,
    account_balance: float,
    account_currency: str
) -> bool:
    """
    Encrypt and store MT5 credentials for a user.
    Only the password is encrypted; login and server are stored plaintext
    to allow account lookup and display without decryption.

    Args:
        telegram_id:      Telegram user ID
        login:            MT5 account login number
        password:         MT5 account password (will be encrypted)
        server:           MT5 broker server name
        broker_name:      Display name of broker
        account_balance:  Current account balance
        account_currency: Account currency code

    Returns:
        bool: True if saved successfully
    """
    try:
        encrypted_password = encrypt_value(password)

        updates = {
            'mt5_login':            login,
            'mt5_password_encrypted': encrypted_password,
            'mt5_server':           server,
            'mt5_broker_name':      broker_name,
            'mt5_account_balance':  account_balance,
            'mt5_account_currency': account_currency,
            'mt5_connected':        True
        }

        result = update_user(telegram_id, updates)

        if result:
            logger.info("Saved MT5 credentials for user %d (login=%d)", telegram_id, login)
            return True

        return False

    except Exception as e:
        logger.error("Error saving MT5 credentials for user %d: %s", telegram_id, e)
        return False


def get_mt5_credentials(telegram_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve and decrypt MT5 credentials for a user.

    Args:
        telegram_id: Telegram user ID

    Returns:
        dict with 'login', 'password', 'server', 'broker_name', or None
    """
    try:
        user = get_user(telegram_id)

        if not user or not user.get('mt5_connected'):
            return None

        encrypted_password = user.get('mt5_password_encrypted')
        if not encrypted_password:
            return None

        try:
            password = decrypt_value(encrypted_password)
        except InvalidToken:
            logger.error(
                "Failed to decrypt MT5 password for user %d. "
                "Encryption key may have changed.", telegram_id
            )
            return None

        return {
            'login':      user['mt5_login'],
            'password':   password,
            'server':     user['mt5_server'],
            'broker_name': user.get('mt5_broker_name', 'Unknown Broker')
        }

    except Exception as e:
        logger.error("Error fetching MT5 credentials for user %d: %s", telegram_id, e)
        return None


def delete_mt5_credentials(telegram_id: int) -> bool:
    """
    Remove MT5 credentials and disconnect user.

    Args:
        telegram_id: Telegram user ID

    Returns:
        bool: True if successful
    """
    try:
        updates = {
            'mt5_login':              None,
            'mt5_password_encrypted': None,
            'mt5_server':             None,
            'mt5_broker_name':        None,
            'mt5_account_balance':    None,
            'mt5_account_currency':   None,
            'mt5_connected':          False,
            'symbol_mappings':        None
        }

        result = update_user(telegram_id, updates)

        if result:
            logger.info("Deleted MT5 credentials for user %d", telegram_id)
            return True
        return False

    except Exception as e:
        logger.error("Error deleting MT5 credentials for user %d: %s", telegram_id, e)
        return False


def save_symbol_mapping(telegram_id: int, standard_symbol: str, broker_symbol: str) -> bool:
    """
    Persist a broker-specific symbol mapping.

    Args:
        telegram_id:     Telegram user ID
        standard_symbol: Canonical symbol (e.g., 'EURUSD')
        broker_symbol:   Broker variant (e.g., 'EURUSD.pro')

    Returns:
        bool: True if saved
    """
    try:
        user = get_user(telegram_id)
        if not user:
            return False

        mappings = user.get('symbol_mappings') or {}
        mappings[standard_symbol] = broker_symbol

        result = update_user(telegram_id, {'symbol_mappings': mappings})
        if result:
            logger.info(
                "Symbol mapping saved for user %d: %s -> %s",
                telegram_id, standard_symbol, broker_symbol
            )
            return True
        return False

    except Exception as e:
        logger.error("Error saving symbol mapping: %s", e)
        return False


def get_symbol_mapping(telegram_id: int, standard_symbol: str) -> Optional[str]:
    """
    Retrieve broker-specific symbol for a user.

    Args:
        telegram_id:     Telegram user ID
        standard_symbol: Canonical symbol

    Returns:
        str: Broker symbol, or None if no mapping exists
    """
    try:
        user = get_user(telegram_id)
        if not user:
            return None
        mappings = user.get('symbol_mappings') or {}
        return mappings.get(standard_symbol)

    except Exception as e:
        logger.error("Error getting symbol mapping: %s", e)
        return None


# ==================== SIGNAL / SETUP MANAGEMENT ====================

def get_next_signal_number() -> int:
    """
    Return the next sequential setup number.

    Returns:
        int: Next signal number
    """
    try:
        client = get_client()
        response = (
            client.table('signals')
            .select('signal_number')
            .order('signal_number', desc=True)
            .limit(1)
            .execute()
        )
        if response.data:
            return response.data[0]['signal_number'] + 1
        return 1

    except Exception as e:
        logger.error("Error getting next signal number: %s", e)
        return 1


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
    lstm_score: int,
    xgboost_score: int,
    session: str,
    order_type: str
) -> Optional[Dict[str, Any]]:
    """
    Save a new automated setup to the database.

    Returns:
        dict: Saved signal record, or None if failed
    """
    try:
        client = get_client()
        signal_number = get_next_signal_number()

        signal_data = {
            'signal_number':  signal_number,
            'symbol':         symbol,
            'direction':      direction,
            'setup_type':     setup_type,
            'entry_price':    entry_price,
            'stop_loss':      stop_loss,
            'take_profit_1':  take_profit_1,
            'take_profit_2':  take_profit_2,
            'sl_pips':        sl_pips,
            'tp1_pips':       tp1_pips,
            'tp2_pips':       tp2_pips,
            'rr_tp1':         rr_tp1,
            'rr_tp2':         rr_tp2,
            'ml_score':       ml_score,
            'lstm_score':     lstm_score,
            'xgboost_score':  xgboost_score,
            'session':        session,
            'order_type':     order_type,
            'created_at':     utils.get_current_utc_time().isoformat()
        }

        response = client.table('signals').insert(signal_data).execute()

        if response.data:
            logger.info("Saved setup #%d: %s %s", signal_number, symbol, direction)
            return response.data[0]

        return None

    except Exception as e:
        logger.error("Error saving signal: %s", e)
        return None


def get_latest_signal() -> Optional[Dict[str, Any]]:
    """
    Retrieve the most recent setup from the last 24 hours.

    Returns:
        dict: Signal record, or None if none found
    """
    try:
        client = get_client()
        cutoff = (utils.get_current_utc_time() - timedelta(hours=24)).isoformat()

        response = (
            client.table('signals')
            .select('*')
            .gte('created_at', cutoff)
            .order('created_at', desc=True)
            .limit(1)
            .execute()
        )

        return response.data[0] if response.data else None

    except Exception as e:
        logger.error("Error fetching latest signal: %s", e)
        return None


# ==================== TRADE MANAGEMENT ====================

def save_trade(
    telegram_id: int,
    signal_id: int,
    mt5_ticket: int,
    symbol: str,
    direction: str,
    lot_size: float,
    entry_price: float,
    stop_loss: float,
    take_profit_1: float,
    take_profit_2: float,
    order_type: str
) -> Optional[Dict[str, Any]]:
    """
    Save an executed trade to the database.

    Returns:
        dict: Saved trade record, or None if failed
    """
    try:
        client = get_client()
        now = utils.get_current_utc_time().isoformat()

        trade_data = {
            'telegram_id':   telegram_id,
            'signal_id':     signal_id,
            'mt5_ticket':    mt5_ticket,
            'symbol':        symbol,
            'direction':     direction,
            'lot_size':      lot_size,
            'entry_price':   entry_price,
            'stop_loss':     stop_loss,
            'take_profit_1': take_profit_1,
            'take_profit_2': take_profit_2,
            'order_type':    order_type,
            'status':        'OPEN',
            'tp1_hit':       False,
            'breakeven_set': False,
            'realized_pnl':  None,
            'rr_achieved':   None,
            'opened_at':     now,
            'created_at':    now
        }

        response = client.table('trades').insert(trade_data).execute()

        if response.data:
            logger.info(
                "Trade saved: user=%d ticket=%d %s %s",
                telegram_id, mt5_ticket, symbol, direction
            )
            return response.data[0]

        return None

    except Exception as e:
        logger.error("Error saving trade: %s", e)
        return None


def update_trade(trade_id: int, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Update fields on an existing trade record.

    Args:
        trade_id: Database primary key of the trade
        updates:  Fields to update

    Returns:
        dict: Updated trade record, or None if failed
    """
    try:
        client = get_client()
        updates['updated_at'] = utils.get_current_utc_time().isoformat()

        response = (
            client.table('trades')
            .update(updates)
            .eq('id', trade_id)
            .execute()
        )

        return response.data[0] if response.data else None

    except Exception as e:
        logger.error("Error updating trade %d: %s", trade_id, e)
        return None


def get_open_trades(telegram_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieve all open trades, optionally filtered by user.

    Args:
        telegram_id: Optional Telegram user ID filter

    Returns:
        list: Open trade records
    """
    try:
        client = get_client()
        query = client.table('trades').select('*').eq('status', 'OPEN')

        if telegram_id is not None:
            query = query.eq('telegram_id', telegram_id)

        response = query.execute()
        return response.data if response.data else []

    except Exception as e:
        logger.error("Error fetching open trades: %s", e)
        return []


# ==================== NEWS EVENTS ====================

def save_news_event(
    event_time_utc: datetime,
    currency: str,
    event_name: str,
    impact: str,
    forecast: Optional[str] = None,
    previous: Optional[str] = None
) -> bool:
    """Save a high-impact news event."""
    try:
        client = get_client()

        event_data = {
            'event_time_utc': event_time_utc.isoformat(),
            'currency':       currency,
            'event_name':     event_name,
            'impact':         impact,
            'forecast':       forecast,
            'previous':       previous,
            'created_at':     utils.get_current_utc_time().isoformat()
        }

        response = client.table('news_events').insert(event_data).execute()
        return bool(response.data)

    except Exception as e:
        logger.error("Error saving news event: %s", e)
        return False


def get_todays_news() -> List[Dict[str, Any]]:
    """
    Retrieve high-impact news events scheduled for today (UTC).

    Returns:
        list: News event records
    """
    try:
        client = get_client()
        today_start = utils.get_current_utc_time().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        tomorrow_start = today_start + timedelta(days=1)

        response = (
            client.table('news_events')
            .select('*')
            .gte('event_time_utc', today_start.isoformat())
            .lt('event_time_utc', tomorrow_start.isoformat())
            .order('event_time_utc')
            .execute()
        )

        return response.data if response.data else []

    except Exception as e:
        logger.error("Error fetching today's news: %s", e)
        return []


def clear_old_news() -> bool:
    """Delete news events older than 7 days."""
    try:
        client = get_client()
        cutoff = (utils.get_current_utc_time() - timedelta(days=7)).isoformat()
        client.table('news_events').delete().lt('event_time_utc', cutoff).execute()
        return True

    except Exception as e:
        logger.error("Error clearing old news: %s", e)
        return False


# ==================== ML MODEL METRICS ====================

def log_model_metrics(
    model_name: str,
    metric_type: str,
    metric_value: float,
    dataset_size: int
) -> bool:
    """
    Log ML model performance metrics for monitoring.

    Args:
        model_name:   e.g. 'LSTM', 'XGBoost'
        metric_type:  e.g. 'accuracy', 'precision', 'f1'
        metric_value: Numeric metric value
        dataset_size: Number of samples used

    Returns:
        bool: True if logged successfully
    """
    try:
        client = get_client()

        data = {
            'model_name':   model_name,
            'metric_type':  metric_type,
            'metric_value': metric_value,
            'dataset_size': dataset_size,
            'timestamp':    utils.get_current_utc_time().isoformat()
        }

        response = client.table('model_metrics').insert(data).execute()
        return bool(response.data)

    except Exception as e:
        logger.error("Error logging model metrics: %s", e)
        return False


# ==================== MESSAGE QUEUE (OFFLINE USER SUPPORT) ====================

def queue_message(
    telegram_id: int,
    message_text: str,
    message_type: str = 'SYSTEM_MESSAGE'
) -> bool:
    """
    Queue a message for a user (for when they're offline).
    Messages will be sent when the user comes back online.

    Args:
        telegram_id:   Telegram user ID
        message_text:  The message content
        message_type:  'SETUP_ALERT', 'TRADE_NOTIFICATION', or 'SYSTEM_MESSAGE'

    Returns:
        bool: True if queued successfully
    """
    try:
        client = get_client()

        data = {
            'telegram_id':  telegram_id,
            'message_text': message_text,
            'message_type': message_type,
            'status':       'PENDING',
            'created_at':   utils.get_current_utc_time().isoformat()
        }

        response = client.table('message_queue').insert(data).execute()
        
        if response.data:
            logger.info(
                "Queued message for user %d (type: %s, timestamp: %s)",
                telegram_id, message_type, data['created_at']
            )
            return True
        return False

    except Exception as e:
        logger.error("Error queuing message for user %d: %s", telegram_id, e)
        return False


def get_pending_messages(telegram_id: int) -> List[Dict[str, Any]]:
    """
    Retrieve all pending queued messages for a user.

    Args:
        telegram_id: Telegram user ID

    Returns:
        list: Pending message records ordered by created_at
    """
    try:
        client = get_client()

        response = (
            client.table('message_queue')
            .select('*')
            .eq('telegram_id', telegram_id)
            .eq('status', 'PENDING')
            .order('created_at')
            .execute()
        )

        return response.data if response.data else []

    except Exception as e:
        logger.error("Error fetching pending messages for user %d: %s", telegram_id, e)
        return []


def mark_message_sent(message_id: int) -> bool:
    """
    Mark a queued message as sent.

    Args:
        message_id: Database ID of the message

    Returns:
        bool: True if updated successfully
    """
    try:
        client = get_client()

        updates = {
            'status':  'SENT',
            'sent_at': utils.get_current_utc_time().isoformat()
        }

        response = (
            client.table('message_queue')
            .update(updates)
            .eq('id', message_id)
            .execute()
        )

        return bool(response.data)

    except Exception as e:
        logger.error("Error marking message %d as sent: %s", message_id, e)
        return False


def clear_old_messages(days: int = 7) -> bool:
    """
    Delete sent messages older than specified days.

    Args:
        days: Delete messages older than this many days

    Returns:
        bool: True if cleanup successful
    """
    try:
        client = get_client()
        cutoff = (utils.get_current_utc_time() - timedelta(days=days)).isoformat()

        client.table('message_queue').delete().eq('status', 'SENT').lt('sent_at', cutoff).execute()
        return True

    except Exception as e:
        logger.error("Error clearing old queued messages: %s", e)
        return False