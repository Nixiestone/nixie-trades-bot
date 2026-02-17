"""
Database operations for Nix Trades Telegram Bot
Supabase PostgreSQL integration
NO EMOJIS - Professional code only
"""

import logging
import os
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from dotenv import load_dotenv
from supabase import create_client, Client
from cryptography.fernet import Fernet
import config
import utils

# Load .env file explicitly so bot.py, database.py, and all modules see it
load_dotenv()

logger = logging.getLogger(__name__)

# Global Supabase client
supabase_client: Optional[Client] = None

# Encryption key for MT5 passwords
encryption_key = None
fernet = None


def init_supabase() -> Client:
    """
    Initialize Supabase client connection.
    Supports both old JWT key format and new sb_publishable_ key format.
    
    Returns:
        Client: Supabase client instance
        
    Raises:
        ValueError: If environment variables are missing
    """
    global supabase_client, encryption_key, fernet
    
    try:
        # Get credentials from environment
        supabase_url = os.getenv('SUPABASE_URL', '').strip()
        supabase_key = os.getenv('SUPABASE_KEY', '').strip()
        encryption_key_env = os.getenv('ENCRYPTION_KEY', '').strip()
        
        # Validate URL
        if not supabase_url:
            raise ValueError(
                "SUPABASE_URL is missing from your .env file.\n"
                "Find it at: Supabase Dashboard → Settings → Data API → Project URL"
            )
        
        if not supabase_url.startswith('https://'):
            raise ValueError(
                f"SUPABASE_URL looks wrong: '{supabase_url}'\n"
                "It must start with https:// and end in .supabase.co"
            )
        
        # Validate key
        if not supabase_key:
            raise ValueError(
                "SUPABASE_KEY is missing from your .env file.\n"
                "Find it at: Supabase Dashboard → Settings → Data API → Project API keys → Publishable"
            )
        
        # Validate encryption key
        if not encryption_key_env:
            raise ValueError(
                "ENCRYPTION_KEY is missing from your .env file.\n"
                "Generate one by running:\n"
                "python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )
        
        # Initialize Supabase client
        # create_client works with both old eyJhbGci... and new sb_publishable_... keys
        supabase_client = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully")
        
        # Initialize encryption
        encryption_key = encryption_key_env.encode()
        fernet = Fernet(encryption_key)
        logger.info("Encryption initialized successfully")
        
        return supabase_client
    
    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {e}")
        raise


def get_client() -> Client:
    """
    Get Supabase client, initializing if necessary.
    
    Returns:
        Client: Supabase client instance
    """
    global supabase_client
    
    if supabase_client is None:
        supabase_client = init_supabase()
    
    return supabase_client


# ==================== USER MANAGEMENT ====================

def create_user(
    telegram_id: int,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
    timezone: str = 'UTC'
) -> Optional[Dict[str, Any]]:
    """
    Create new user in database.
    
    Args:
        telegram_id: Telegram user ID
        username: Telegram username (without @)
        first_name: User's first name
        timezone: IANA timezone string
        
    Returns:
        dict: Created user record, or None if failed
    """
    try:
        client = get_client()
        
        user_data = {
            'telegram_id': telegram_id,
            'username': username,
            'first_name': first_name,
            'timezone': timezone,
            'subscription_status': 'inactive',
            'risk_percent': config.DEFAULT_RISK_PERCENT,
            'created_at': utils.get_current_utc_time().isoformat(),
            'updated_at': utils.get_current_utc_time().isoformat()
        }
        
        response = client.table('telegram_users').insert(user_data).execute()
        
        if response.data:
            logger.info(f"Created user: telegram_id={telegram_id}, timezone={timezone}")
            return response.data[0]
        else:
            logger.error(f"Failed to create user: {response}")
            return None
    
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return None


def get_user(telegram_id: int) -> Optional[Dict[str, Any]]:
    """
    Get user by Telegram ID.
    
    Args:
        telegram_id: Telegram user ID
        
    Returns:
        dict: User record, or None if not found
    """
    try:
        client = get_client()
        
        response = client.table('telegram_users').select('*').eq('telegram_id', telegram_id).execute()
        
        if response.data:
            return response.data[0]
        else:
            return None
    
    except Exception as e:
        logger.error(f"Error fetching user {telegram_id}: {e}")
        return None


def get_or_create_user(
    telegram_id: int,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
    timezone: str = 'UTC'
) -> Optional[Dict[str, Any]]:
    """
    Get existing user or create if not exists.
    
    Args:
        telegram_id: Telegram user ID
        username: Telegram username
        first_name: User's first name
        timezone: IANA timezone string
        
    Returns:
        dict: User record
    """
    user = get_user(telegram_id)
    
    if user is None:
        user = create_user(telegram_id, username, first_name, timezone)
    
    return user


def update_user(telegram_id: int, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Update user fields.
    
    Args:
        telegram_id: Telegram user ID
        updates: Dictionary of fields to update
        
    Returns:
        dict: Updated user record, or None if failed
        
    Example:
        >>> update_user(123456789, {'subscription_status': 'active', 'timezone': 'America/New_York'})
    """
    try:
        client = get_client()
        
        # Add updated_at timestamp
        updates['updated_at'] = utils.get_current_utc_time().isoformat()
        
        response = client.table('telegram_users').update(updates).eq('telegram_id', telegram_id).execute()
        
        if response.data:
            logger.info(f"Updated user {telegram_id}: {list(updates.keys())}")
            return response.data[0]
        else:
            logger.error(f"Failed to update user {telegram_id}")
            return None
    
    except Exception as e:
        logger.error(f"Error updating user {telegram_id}: {e}")
        return None


def activate_subscription(telegram_id: int) -> bool:
    """
    Activate user subscription.
    
    Args:
        telegram_id: Telegram user ID
        
    Returns:
        bool: True if successful
    """
    try:
        updates = {
            'subscription_status': 'active',
            'trial_started_at': utils.get_current_utc_time().isoformat()
        }
        
        result = update_user(telegram_id, updates)
        return result is not None
    
    except Exception as e:
        logger.error(f"Error activating subscription for {telegram_id}: {e}")
        return False


def deactivate_subscription(telegram_id: int) -> bool:
    """
    Deactivate user subscription.
    
    Args:
        telegram_id: Telegram user ID
        
    Returns:
        bool: True if successful
    """
    try:
        updates = {'subscription_status': 'inactive'}
        result = update_user(telegram_id, updates)
        return result is not None
    
    except Exception as e:
        logger.error(f"Error deactivating subscription for {telegram_id}: {e}")
        return False


def update_timezone(telegram_id: int, timezone: str) -> bool:
    """
    Update user timezone.
    
    Args:
        telegram_id: Telegram user ID
        timezone: IANA timezone string
        
    Returns:
        bool: True if successful
    """
    try:
        updates = {'timezone': timezone}
        result = update_user(telegram_id, updates)
        return result is not None
    
    except Exception as e:
        logger.error(f"Error updating timezone for {telegram_id}: {e}")
        return False


def update_risk_percent(telegram_id: int, risk_percent: float) -> bool:
    """
    Update user risk percentage.
    
    Args:
        telegram_id: Telegram user ID
        risk_percent: Risk per trade (0.1 - 5.0)
        
    Returns:
        bool: True if successful
    """
    try:
        if not utils.validate_risk_percent(risk_percent):
            logger.error(f"Invalid risk percent: {risk_percent}")
            return False
        
        updates = {'risk_percent': risk_percent}
        result = update_user(telegram_id, updates)
        return result is not None
    
    except Exception as e:
        logger.error(f"Error updating risk percent for {telegram_id}: {e}")
        return False


def update_last_8am_alert(telegram_id: int) -> bool:
    """
    Update timestamp of last 8 AM alert sent.
    
    Args:
        telegram_id: Telegram user ID
        
    Returns:
        bool: True if successful
    """
    try:
        updates = {'last_8am_alert_sent': utils.get_current_utc_time().isoformat()}
        result = update_user(telegram_id, updates)
        return result is not None
    
    except Exception as e:
        logger.error(f"Error updating last_8am_alert for {telegram_id}: {e}")
        return False


def get_active_subscribers() -> List[Dict[str, Any]]:
    """
    Get all users with active subscriptions.
    
    Returns:
        List[dict]: List of active user records
    """
    try:
        client = get_client()
        
        response = client.table('telegram_users').select('*').eq('subscription_status', 'active').execute()
        
        if response.data:
            logger.info(f"Found {len(response.data)} active subscribers")
            return response.data
        else:
            return []
    
    except Exception as e:
        logger.error(f"Error fetching active subscribers: {e}")
        return []


# ==================== MT5 CREDENTIALS ====================

def save_mt5_credentials(
    telegram_id: int,
    mt5_login: str,
    mt5_password: str,
    mt5_server: str,
    mt5_broker_name: str
) -> bool:
    """
    Save encrypted MT5 credentials.
    
    Args:
        telegram_id: Telegram user ID
        mt5_login: MT5 account login number
        mt5_password: MT5 account password (will be encrypted)
        mt5_server: MT5 server name
        mt5_broker_name: Broker display name
        
    Returns:
        bool: True if successful
        
    Security:
        Password is encrypted with Fernet before storage
    """
    try:
        global fernet
        
        if fernet is None:
            logger.error("Encryption not initialized")
            return False
        
        # Encrypt password
        encrypted_password = fernet.encrypt(mt5_password.encode()).decode()
        
        updates = {
            'mt5_login': mt5_login,
            'mt5_password_encrypted': encrypted_password,
            'mt5_server': mt5_server,
            'mt5_broker_name': mt5_broker_name,
            'mt5_connected': True
        }
        
        result = update_user(telegram_id, updates)
        
        if result:
            logger.info(f"Saved MT5 credentials for user {telegram_id} (broker: {mt5_broker_name})")
            return True
        else:
            return False
    
    except Exception as e:
        logger.error(f"Error saving MT5 credentials: {e}")
        return False


def get_mt5_credentials(telegram_id: int) -> Optional[Dict[str, str]]:
    """
    Get decrypted MT5 credentials.
    
    Args:
        telegram_id: Telegram user ID
        
    Returns:
        dict: {'login': str, 'password': str, 'server': str, 'broker_name': str}
        or None if not found or decryption fails
    """
    try:
        global fernet
        
        if fernet is None:
            logger.error("Encryption not initialized")
            return None
        
        user = get_user(telegram_id)
        
        if not user or not user.get('mt5_connected'):
            return None
        
        # Decrypt password
        encrypted_password = user.get('mt5_password_encrypted')
        if not encrypted_password:
            return None
        
        decrypted_password = fernet.decrypt(encrypted_password.encode()).decode()
        
        return {
            'login': user.get('mt5_login'),
            'password': decrypted_password,
            'server': user.get('mt5_server'),
            'broker_name': user.get('mt5_broker_name')
        }
    
    except Exception as e:
        logger.error(f"Error retrieving MT5 credentials: {e}")
        return None


def delete_mt5_credentials(telegram_id: int) -> bool:
    """
    Delete MT5 credentials (disconnect).
    
    Args:
        telegram_id: Telegram user ID
        
    Returns:
        bool: True if successful
    """
    try:
        updates = {
            'mt5_login': None,
            'mt5_password_encrypted': None,
            'mt5_server': None,
            'mt5_broker_name': None,
            'mt5_connected': False,
            'symbol_mappings': None
        }
        
        result = update_user(telegram_id, updates)
        
        if result:
            logger.info(f"Deleted MT5 credentials for user {telegram_id}")
            return True
        else:
            return False
    
    except Exception as e:
        logger.error(f"Error deleting MT5 credentials: {e}")
        return False


def save_symbol_mapping(telegram_id: int, standard_symbol: str, broker_symbol: str) -> bool:
    """
    Save symbol mapping for user's broker.
    
    Args:
        telegram_id: Telegram user ID
        standard_symbol: Standard symbol (e.g., 'EURUSD')
        broker_symbol: Broker's symbol variant (e.g., 'EURUSD.pro')
        
    Returns:
        bool: True if successful
    """
    try:
        user = get_user(telegram_id)
        
        if not user:
            return False
        
        # Get existing mappings or create new dict
        mappings = user.get('symbol_mappings') or {}
        
        # Add new mapping
        mappings[standard_symbol] = broker_symbol
        
        updates = {'symbol_mappings': mappings}
        result = update_user(telegram_id, updates)
        
        if result:
            logger.info(f"Saved symbol mapping for user {telegram_id}: {standard_symbol} -> {broker_symbol}")
            return True
        else:
            return False
    
    except Exception as e:
        logger.error(f"Error saving symbol mapping: {e}")
        return False


def get_symbol_mapping(telegram_id: int, standard_symbol: str) -> Optional[str]:
    """
    Get broker-specific symbol mapping.
    
    Args:
        telegram_id: Telegram user ID
        standard_symbol: Standard symbol (e.g., 'EURUSD')
        
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
        logger.error(f"Error getting symbol mapping: {e}")
        return None


# ==================== SIGNAL MANAGEMENT ====================

def get_next_signal_number() -> int:
    """
    Get next available signal number (auto-increment).
    
    Returns:
        int: Next signal number
    """
    try:
        client = get_client()
        
        response = client.table('signals').select('signal_number').order('signal_number', desc=True).limit(1).execute()
        
        if response.data:
            return response.data[0]['signal_number'] + 1
        else:
            return 1
    
    except Exception as e:
        logger.error(f"Error getting next signal number: {e}")
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
    """Save new automated setup to database."""
    try:
        client = get_client()
        
        signal_number = get_next_signal_number()
        
        signal_data = {
            'signal_number': signal_number,
            'symbol': symbol,
            'direction': direction,
            'setup_type': setup_type,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit_1': take_profit_1,
            'take_profit_2': take_profit_2,
            'sl_pips': sl_pips,
            'tp1_pips': tp1_pips,
            'tp2_pips': tp2_pips,
            'rr_tp1': rr_tp1,
            'rr_tp2': rr_tp2,
            'ml_score': ml_score,
            'lstm_score': lstm_score,
            'xgboost_score': xgboost_score,
            'session': session,
            'order_type': order_type,
            'created_at': utils.get_current_utc_time().isoformat()
        }
        
        response = client.table('signals').insert(signal_data).execute()
        
        if response.data:
            logger.info(f"Saved signal #{signal_number}: {symbol} {direction}")
            return response.data[0]
        else:
            return None
    
    except Exception as e:
        logger.error(f"Error saving signal: {e}")
        return None


def get_latest_signal() -> Optional[Dict[str, Any]]:
    """Get most recent signal from last 24 hours."""
    try:
        client = get_client()
        cutoff = (utils.get_current_utc_time() - timedelta(hours=24)).isoformat()
        
        response = client.table('signals').select('*').gte('created_at', cutoff).order('created_at', desc=True).limit(1).execute()
        
        return response.data[0] if response.data else None
    
    except Exception as e:
        logger.error(f"Error fetching latest signal: {e}")
        return None


# ==================== NEWS EVENTS ====================

def save_news_event(
    event_time_utc: datetime,
    currency: str,
    event_name: str,
    impact: str,
    forecast: Optional[str] = None,
    previous: Optional[str] = None
) -> bool:
    """Save high-impact news event."""
    try:
        client = get_client()
        
        event_data = {
            'event_time_utc': event_time_utc.isoformat(),
            'currency': currency,
            'event_name': event_name,
            'impact': impact,
            'forecast': forecast,
            'previous': previous,
            'created_at': utils.get_current_utc_time().isoformat()
        }
        
        response = client.table('news_events').insert(event_data).execute()
        return bool(response.data)
    
    except Exception as e:
        logger.error(f"Error saving news event: {e}")
        return False


def get_todays_news() -> List[Dict[str, Any]]:
    """Get today's cached high-impact news events."""
    try:
        client = get_client()
        
        today_start = utils.get_current_utc_time().replace(hour=0, minute=0, second=0)
        tomorrow_start = today_start + timedelta(days=1)
        
        response = client.table('news_events').select('*').gte('event_time_utc', today_start.isoformat()).lt('event_time_utc', tomorrow_start.isoformat()).order('event_time_utc').execute()
        
        return response.data if response.data else []
    
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return []


def clear_old_news() -> bool:
    """Delete news events older than 7 days."""
    try:
        client = get_client()
        cutoff = (utils.get_current_utc_time() - timedelta(days=7)).isoformat()
        
        response = client.table('news_events').delete().lt('event_time_utc', cutoff).execute()
        return True
    
    except Exception as e:
        logger.error(f"Error clearing old news: {e}")
        return False


# ==================== ML MODEL METRICS ====================

def log_model_metrics(
    model_name: str,
    metric_type: str,
    metric_value: float,
    dataset_size: int
) -> bool:
    """Log ML model performance metrics."""
    try:
        client = get_client()
        
        data = {
            'model_name': model_name,
            'metric_type': metric_type,
            'metric_value': metric_value,
            'dataset_size': dataset_size,
            'timestamp': utils.get_current_utc_time().isoformat()
        }
        
        response = client.table('model_metrics').insert(data).execute()
        return bool(response.data)
    
    except Exception as e:
        logger.error(f"Error logging model metrics: {e}")
        return False