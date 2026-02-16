"""
NIXIE LOGIC - Database Operations
Supabase PostgreSQL database layer
NO EMOJIS | Production-grade error handling

Author: NIXIE LOGIC Development Team
Contact: support@nixielogic.com
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List
from cryptography.fernet import Fernet
from supabase import create_client, Client
import config
import utils

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase: Optional[Client] = None


# ==================== INITIALIZATION ====================
def init_supabase() -> Client:
    """
    Initialize Supabase client connection.
    
    Returns:
        Supabase client instance
        
    Raises:
        Exception if connection fails
    """
    global supabase
    
    try:
        if not config.SUPABASE_URL or not config.SUPABASE_KEY:
            raise ValueError("Supabase credentials not configured in environment")
        
        supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        logger.info("Supabase client initialized successfully")
        return supabase
        
    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {e}")
        raise


def get_supabase() -> Client:
    """
    Get Supabase client instance (initialize if needed).
    
    Returns:
        Supabase client
    """
    global supabase
    if supabase is None:
        supabase = init_supabase()
    return supabase


# ==================== ENCRYPTION HELPERS ====================
def _get_cipher() -> Fernet:
    """
    Get Fernet cipher for MT5 credential encryption.
    
    Returns:
        Fernet cipher instance
    """
    if not config.ENCRYPTION_KEY:
        raise ValueError("Encryption key not configured")
    
    # Ensure key is properly formatted
    key = config.ENCRYPTION_KEY.encode() if isinstance(config.ENCRYPTION_KEY, str) else config.ENCRYPTION_KEY
    return Fernet(key)


def _encrypt_password(password: str) -> str:
    """
    Encrypt MT5 password before storage.
    
    Args:
        password: Plain text password
        
    Returns:
        Encrypted password as string
    """
    cipher = _get_cipher()
    encrypted = cipher.encrypt(password.encode())
    return encrypted.decode()


def _decrypt_password(encrypted_password: str) -> str:
    """
    Decrypt MT5 password from storage.
    
    Args:
        encrypted_password: Encrypted password string
        
    Returns:
        Plain text password
    """
    cipher = _get_cipher()
    decrypted = cipher.decrypt(encrypted_password.encode())
    return decrypted.decode()


# ==================== USER OPERATIONS ====================
def create_user(
    telegram_id: int,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
    timezone: str = config.DEFAULT_TIMEZONE
) -> Dict[str, Any]:
    """
    Create new user in database.
    
    Args:
        telegram_id: Telegram user ID
        username: Telegram username (optional)
        first_name: User's first name (optional)
        timezone: User's timezone (IANA string)
        
    Returns:
        Created user record
    """
    try:
        client = get_supabase()
        
        user_data = {
            "telegram_id": telegram_id,
            "username": username,
            "first_name": first_name,
            "timezone": timezone,
            "subscription_status": "inactive",
            "risk_percent": float(config.DEFAULT_RISK_PERCENT),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = client.table(config.TABLE_USERS).insert(user_data).execute()
        
        logger.info(f"User created: telegram_id={telegram_id}")
        utils.log_user_action(telegram_id, "user_created")
        
        return response.data[0] if response.data else user_data
        
    except Exception as e:
        logger.error(f"Error creating user {telegram_id}: {e}")
        raise


def get_user(telegram_id: int) -> Optional[Dict[str, Any]]:
    """
    Get user by Telegram ID.
    
    Args:
        telegram_id: Telegram user ID
        
    Returns:
        User record or None if not found
    """
    try:
        client = get_supabase()
        
        response = client.table(config.TABLE_USERS).select("*").eq(
            "telegram_id", telegram_id
        ).execute()
        
        if response.data:
            return response.data[0]
        else:
            logger.warning(f"User not found: telegram_id={telegram_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching user {telegram_id}: {e}")
        return None


def get_or_create_user(
    telegram_id: int,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
    timezone: str = config.DEFAULT_TIMEZONE
) -> Dict[str, Any]:
    """
    Get existing user or create new one.
    
    Args:
        telegram_id: Telegram user ID
        username: Telegram username
        first_name: User's first name
        timezone: User's timezone
        
    Returns:
        User record
    """
    user = get_user(telegram_id)
    
    if user is None:
        user = create_user(telegram_id, username, first_name, timezone)
    
    return user


def update_user(telegram_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update user record.
    
    Args:
        telegram_id: Telegram user ID
        updates: Dictionary of fields to update
        
    Returns:
        Updated user record
    """
    try:
        client = get_supabase()
        
        updates["updated_at"] = datetime.utcnow().isoformat()
        
        response = client.table(config.TABLE_USERS).update(updates).eq(
            "telegram_id", telegram_id
        ).execute()
        
        logger.info(f"User updated: telegram_id={telegram_id}, fields={list(updates.keys())}")
        utils.log_user_action(telegram_id, "user_updated", updates)
        
        return response.data[0] if response.data else {}
        
    except Exception as e:
        logger.error(f"Error updating user {telegram_id}: {e}")
        raise


def activate_subscription(telegram_id: int) -> Dict[str, Any]:
    """
    Activate user subscription.
    
    Args:
        telegram_id: Telegram user ID
        
    Returns:
        Updated user record
    """
    updates = {
        "subscription_status": "active",
        "trial_started_at": datetime.utcnow().isoformat()
    }
    
    utils.log_user_action(telegram_id, "subscription_activated")
    return update_user(telegram_id, updates)


def deactivate_subscription(telegram_id: int) -> Dict[str, Any]:
    """
    Deactivate user subscription.
    
    Args:
        telegram_id: Telegram user ID
        
    Returns:
        Updated user record
    """
    updates = {
        "subscription_status": "inactive"
    }
    
    utils.log_user_action(telegram_id, "subscription_deactivated")
    return update_user(telegram_id, updates)


def get_active_subscribers() -> List[Dict[str, Any]]:
    """
    Get all users with active subscriptions.
    
    Returns:
        List of active user records
    """
    try:
        client = get_supabase()
        
        response = client.table(config.TABLE_USERS).select("*").eq(
            "subscription_status", "active"
        ).execute()
        
        logger.info(f"Retrieved {len(response.data)} active subscribers")
        return response.data
        
    except Exception as e:
        logger.error(f"Error fetching active subscribers: {e}")
        return []


# ==================== MT5 CREDENTIALS ====================
def save_mt5_credentials(
    telegram_id: int,
    login: str,
    password: str,
    server: str,
    broker_name: str
) -> Dict[str, Any]:
    """
    Save encrypted MT5 credentials to database.
    
    Args:
        telegram_id: Telegram user ID
        login: MT5 account login number
        password: MT5 account password (will be encrypted)
        server: MT5 server name
        broker_name: Broker display name
        
    Returns:
        Updated user record
    """
    try:
        encrypted_password = _encrypt_password(password)
        
        updates = {
            "mt5_login": login,
            "mt5_password_encrypted": encrypted_password,
            "mt5_server": server,
            "mt5_broker_name": broker_name,
            "mt5_connected": True
        }
        
        utils.log_user_action(telegram_id, "mt5_credentials_saved", {"broker": broker_name})
        return update_user(telegram_id, updates)
        
    except Exception as e:
        logger.error(f"Error saving MT5 credentials for user {telegram_id}: {e}")
        raise


def get_mt5_credentials(telegram_id: int) -> Optional[Dict[str, str]]:
    """
    Retrieve and decrypt MT5 credentials.
    
    Args:
        telegram_id: Telegram user ID
        
    Returns:
        Dictionary with login, password, server, broker_name or None
    """
    try:
        user = get_user(telegram_id)
        
        if not user or not user.get('mt5_connected'):
            return None
        
        encrypted_password = user.get('mt5_password_encrypted')
        if not encrypted_password:
            return None
        
        password = _decrypt_password(encrypted_password)
        
        return {
            "login": user.get('mt5_login'),
            "password": password,
            "server": user.get('mt5_server'),
            "broker_name": user.get('mt5_broker_name')
        }
        
    except Exception as e:
        logger.error(f"Error retrieving MT5 credentials for user {telegram_id}: {e}")
        return None


def disconnect_mt5(telegram_id: int) -> Dict[str, Any]:
    """
    Remove MT5 credentials and disconnect.
    
    Args:
        telegram_id: Telegram user ID
        
    Returns:
        Updated user record
    """
    updates = {
        "mt5_login": None,
        "mt5_password_encrypted": None,
        "mt5_server": None,
        "mt5_broker_name": None,
        "mt5_connected": False,
        "symbol_mappings": None
    }
    
    utils.log_user_action(telegram_id, "mt5_disconnected")
    return update_user(telegram_id, updates)


# ==================== SIGNAL OPERATIONS ====================
def save_signal(signal_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save new automated setup to database.
    
    Args:
        signal_data: Signal information dictionary
        
    Returns:
        Created signal record
    """
    try:
        client = get_supabase()
        
        # Generate unique signal number
        response = client.table(config.TABLE_SIGNALS).select("signal_number").order(
            "signal_number", desc=True
        ).limit(1).execute()
        
        last_signal_number = response.data[0]['signal_number'] if response.data else 0
        signal_data['signal_number'] = last_signal_number + 1
        
        signal_data['created_at'] = datetime.utcnow().isoformat()
        
        response = client.table(config.TABLE_SIGNALS).insert(signal_data).execute()
        
        logger.info(f"Signal saved: #{signal_data['signal_number']} - {signal_data.get('symbol')} {signal_data.get('direction')}")
        
        return response.data[0] if response.data else signal_data
        
    except Exception as e:
        logger.error(f"Error saving signal: {e}")
        raise


def get_latest_signal(hours: int = 24) -> Optional[Dict[str, Any]]:
    """
    Get most recent signal within specified hours.
    
    Args:
        hours: Number of hours to look back
        
    Returns:
        Latest signal record or None
    """
    try:
        client = get_supabase()
        
        cutoff_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        
        response = client.table(config.TABLE_SIGNALS).select("*").gte(
            "created_at", cutoff_time
        ).order("created_at", desc=True).limit(1).execute()
        
        if response.data:
            return response.data[0]
        else:
            logger.info(f"No signals found in last {hours} hours")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching latest signal: {e}")
        return None


def get_signal_by_number(signal_number: int) -> Optional[Dict[str, Any]]:
    """
    Get signal by signal number.
    
    Args:
        signal_number: Signal number
        
    Returns:
        Signal record or None
    """
    try:
        client = get_supabase()
        
        response = client.table(config.TABLE_SIGNALS).select("*").eq(
            "signal_number", signal_number
        ).execute()
        
        if response.data:
            return response.data[0]
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error fetching signal #{signal_number}: {e}")
        return None


def count_total_signals() -> int:
    """
    Get total number of signals generated.
    
    Returns:
        Total signal count
    """
    try:
        client = get_supabase()
        
        response = client.table(config.TABLE_SIGNALS).select("id", count="exact").execute()
        
        return response.count if hasattr(response, 'count') else 0
        
    except Exception as e:
        logger.error(f"Error counting signals: {e}")
        return 0


# ==================== TRADE OPERATIONS ====================
def save_trade(trade_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save new trade to database.
    
    Args:
        trade_data: Trade information dictionary
        
    Returns:
        Created trade record
    """
    try:
        client = get_supabase()
        
        trade_data['created_at'] = datetime.utcnow().isoformat()
        trade_data['opened_at'] = datetime.utcnow().isoformat()
        
        response = client.table(config.TABLE_TRADES).insert(trade_data).execute()
        
        logger.info(f"Trade saved: User {trade_data.get('user_id')} - {trade_data.get('symbol')} {trade_data.get('direction')}")
        utils.log_trade_event(
            response.data[0]['id'] if response.data else 0,
            "trade_opened",
            {"symbol": trade_data.get('symbol'), "direction": trade_data.get('direction')}
        )
        
        return response.data[0] if response.data else trade_data
        
    except Exception as e:
        logger.error(f"Error saving trade: {e}")
        raise


def update_trade(trade_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update trade record.
    
    Args:
        trade_id: Trade ID
        updates: Dictionary of fields to update
        
    Returns:
        Updated trade record
    """
    try:
        client = get_supabase()
        
        response = client.table(config.TABLE_TRADES).update(updates).eq(
            "id", trade_id
        ).execute()
        
        logger.info(f"Trade updated: ID {trade_id}, fields={list(updates.keys())}")
        utils.log_trade_event(trade_id, "trade_updated", updates)
        
        return response.data[0] if response.data else {}
        
    except Exception as e:
        logger.error(f"Error updating trade {trade_id}: {e}")
        raise


def close_trade(trade_id: int, final_pnl: Decimal, rr_achieved: Decimal) -> Dict[str, Any]:
    """
    Close trade and record final P&L.
    
    Args:
        trade_id: Trade ID
        final_pnl: Final realized profit/loss
        rr_achieved: Risk-reward ratio achieved
        
    Returns:
        Updated trade record
    """
    updates = {
        "status": "CLOSED",
        "closed_at": datetime.utcnow().isoformat(),
        "realized_pnl": float(final_pnl),
        "rr_achieved": float(rr_achieved)
    }
    
    utils.log_trade_event(trade_id, "trade_closed", {"pnl": float(final_pnl), "rr": float(rr_achieved)})
    return update_trade(trade_id, updates)


def get_open_trades(user_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get all open trades (optionally filtered by user).
    
    Args:
        user_id: Optional user ID filter
        
    Returns:
        List of open trade records
    """
    try:
        client = get_supabase()
        
        query = client.table(config.TABLE_TRADES).select("*").eq("status", "OPEN")
        
        if user_id is not None:
            query = query.eq("user_id", user_id)
        
        response = query.execute()
        
        logger.info(f"Retrieved {len(response.data)} open trades" + (f" for user {user_id}" if user_id else ""))
        return response.data
        
    except Exception as e:
        logger.error(f"Error fetching open trades: {e}")
        return []


def get_closed_trades(
    user_id: Optional[int] = None,
    days: int = 30
) -> List[Dict[str, Any]]:
    """
    Get closed trades within specified days.
    
    Args:
        user_id: Optional user ID filter
        days: Number of days to look back
        
    Returns:
        List of closed trade records
    """
    try:
        client = get_supabase()
        
        cutoff_time = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        query = client.table(config.TABLE_TRADES).select("*").eq(
            "status", "CLOSED"
        ).gte("closed_at", cutoff_time)
        
        if user_id is not None:
            query = query.eq("user_id", user_id)
        
        response = query.order("closed_at", desc=True).execute()
        
        logger.info(f"Retrieved {len(response.data)} closed trades in last {days} days")
        return response.data
        
    except Exception as e:
        logger.error(f"Error fetching closed trades: {e}")
        return []


# ==================== USER STATISTICS ====================
def get_user_stats(user_id: int) -> Dict[str, Any]:
    """
    Calculate comprehensive user statistics.
    
    Args:
        user_id: User ID (database ID, not telegram_id)
        
    Returns:
        Statistics dictionary
    """
    try:
        closed_trades = get_closed_trades(user_id=user_id, days=30)
        open_trades = get_open_trades(user_id=user_id)
        
        stats = {
            "total_trades": len(closed_trades),
            "open_trades": len(open_trades),
            "winning_trades": sum(1 for t in closed_trades if Decimal(str(t.get('realized_pnl', 0))) > 0),
            "losing_trades": sum(1 for t in closed_trades if Decimal(str(t.get('realized_pnl', 0))) < 0),
            "total_pnl": sum(Decimal(str(t.get('realized_pnl', 0))) for t in closed_trades),
            "win_rate": utils.calculate_win_rate(closed_trades),
            "avg_rr": utils.calculate_average_rr(closed_trades),
            "sharpe_ratio": utils.calculate_sharpe_ratio(closed_trades)
        }
        
        # Today's P&L
        today_trades = [t for t in closed_trades if t.get('closed_at', '').startswith(datetime.utcnow().date().isoformat())]
        stats["today_pnl"] = sum(Decimal(str(t.get('realized_pnl', 0))) for t in today_trades)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating user stats for user {user_id}: {e}")
        return {}


# ==================== NEWS OPERATIONS ====================
def save_news_events(events: List[Dict[str, Any]]) -> int:
    """
    Save multiple news events to database (bulk insert).
    
    Args:
        events: List of news event dictionaries
        
    Returns:
        Number of events saved
    """
    try:
        if not events:
            return 0
        
        client = get_supabase()
        
        for event in events:
            event['created_at'] = datetime.utcnow().isoformat()
            event['updated_at'] = datetime.utcnow().isoformat()
        
        response = client.table(config.TABLE_NEWS).insert(events).execute()
        
        count = len(response.data) if response.data else 0
        logger.info(f"Saved {count} news events")
        
        return count
        
    except Exception as e:
        logger.error(f"Error saving news events: {e}")
        return 0


def get_todays_news() -> List[Dict[str, Any]]:
    """
    Get today's high-impact news events.
    
    Returns:
        List of news event records
    """
    try:
        client = get_supabase()
        
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        today_end = (datetime.utcnow().replace(hour=23, minute=59, second=59, microsecond=999999)).isoformat()
        
        response = client.table(config.TABLE_NEWS).select("*").gte(
            "event_time_utc", today_start
        ).lte("event_time_utc", today_end).eq(
            "impact", "HIGH"
        ).order("event_time_utc").execute()
        
        logger.info(f"Retrieved {len(response.data)} high-impact news events for today")
        return response.data
        
    except Exception as e:
        logger.error(f"Error fetching today's news: {e}")
        return []


def get_upcoming_news(hours: int = 1) -> List[Dict[str, Any]]:
    """
    Get high-impact news events in the next N hours.
    
    Args:
        hours: Number of hours to look ahead
        
    Returns:
        List of upcoming news events
    """
    try:
        client = get_supabase()
        
        now = datetime.utcnow().isoformat()
        future = (datetime.utcnow() + timedelta(hours=hours)).isoformat()
        
        response = client.table(config.TABLE_NEWS).select("*").gte(
            "event_time_utc", now
        ).lte("event_time_utc", future).eq(
            "impact", "HIGH"
        ).order("event_time_utc").execute()
        
        return response.data
        
    except Exception as e:
        logger.error(f"Error fetching upcoming news: {e}")
        return []


# ==================== MODEL METRICS ====================
def log_model_metrics(
    model_name: str,
    metric_type: str,
    metric_value: Decimal,
    dataset_size: int
) -> Dict[str, Any]:
    """
    Log machine learning model performance metrics.
    
    Args:
        model_name: Model name (e.g., 'LSTM', 'XGBoost')
        metric_type: Metric type (e.g., 'accuracy', 'precision', 'recall')
        metric_value: Metric value (0-100)
        dataset_size: Number of samples in dataset
        
    Returns:
        Created metric record
    """
    try:
        client = get_supabase()
        
        metric_data = {
            "model_name": model_name,
            "metric_type": metric_type,
            "metric_value": float(metric_value),
            "dataset_size": dataset_size,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        response = client.table(config.TABLE_MODEL_METRICS).insert(metric_data).execute()
        
        logger.info(f"Model metric logged: {model_name} {metric_type}={metric_value:.2f}%")
        
        return response.data[0] if response.data else metric_data
        
    except Exception as e:
        logger.error(f"Error logging model metrics: {e}")
        raise


def get_latest_model_metrics(model_name: str) -> Dict[str, Decimal]:
    """
    Get latest metrics for a specific model.
    
    Args:
        model_name: Model name
        
    Returns:
        Dictionary of metric_type: value
    """
    try:
        client = get_supabase()
        
        response = client.table(config.TABLE_MODEL_METRICS).select("*").eq(
            "model_name", model_name
        ).order("timestamp", desc=True).limit(10).execute()
        
        metrics = {}
        for record in response.data:
            metric_type = record.get('metric_type')
            metric_value = Decimal(str(record.get('metric_value', 0)))
            metrics[metric_type] = metric_value
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error fetching model metrics for {model_name}: {e}")
        return {}


# ==================== DATABASE MAINTENANCE ====================
def cleanup_old_data(days: int = 90):
    """
    Clean up old data to manage database size.
    
    Args:
        days: Delete records older than this many days
    """
    try:
        client = get_supabase()
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        # Delete old closed trades
        trades_response = client.table(config.TABLE_TRADES).delete().eq(
            "status", "CLOSED"
        ).lt("closed_at", cutoff_date).execute()
        
        # Delete old news events
        news_response = client.table(config.TABLE_NEWS).delete().lt(
            "event_time_utc", cutoff_date
        ).execute()
        
        logger.info(f"Cleanup completed: Deleted records older than {days} days")
        
    except Exception as e:
        logger.error(f"Error during data cleanup: {e}")


logger.info("NIXIE LOGIC Database Operations Loaded")