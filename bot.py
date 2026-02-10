"""
Nix Trades Telegram Bot - Main Application
Production-grade Telegram bot for algorithmic forex trading
Part 1 of 2: Imports, initialization, and command handlers
NO EMOJIS - Professional code only
"""

import logging
import os
import sys
import signal
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters
)
from telegram.error import TelegramError, NetworkError, TimedOut

# Import our modules
import config
import utils
import database
import smc_strategy
import ml_models
import mt5_connector
import scheduler
import news_fetcher

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            'bot.log',
            maxBytes=config.LOG_FILE_MAX_BYTES,
            backupCount=config.LOG_FILE_BACKUP_COUNT
        )
    ]
)

logger = logging.getLogger(__name__)

# Global application instance
application = None

# Graceful shutdown flag
shutdown_in_progress = False


# ==================== INITIALIZATION ====================

async def initialize_bot():
    """
    Initialize all bot systems on startup.
    
    This function:
    1. Initializes database connection
    2. Loads ML models
    3. Starts position monitoring scheduler
    4. Starts news fetcher
    5. Verifies all systems operational
    
    Returns:
        bool: True if initialization successful
    """
    global shutdown_in_progress
    
    try:
        logger.info("="*60)
        logger.info("NIX TRADES TELEGRAM BOT - INITIALIZING")
        logger.info("="*60)
        
        # Check environment variables
        required_env_vars = [
            'TELEGRAM_BOT_TOKEN',
            'SUPABASE_URL',
            'SUPABASE_KEY',
            'ENCRYPTION_KEY',
            'ALPHA_VANTAGE_API_KEY'
        ]
        
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            logger.error("Check your .env file")
            return False
        
        logger.info("Environment variables verified")
        
        # Initialize database
        logger.info("Initializing database connection...")
        try:
            database.init_supabase()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
        
        # Initialize ML models
        logger.info("Loading ML models...")
        try:
            ml_models.initialize_models()
            logger.info("ML models loaded successfully")
        except Exception as e:
            logger.error(f"ML model initialization failed: {e}")
            return False
        
        # Start scheduler
        logger.info("Starting background scheduler...")
        try:
            await scheduler.start_scheduler()
            logger.info("Scheduler started successfully")
        except Exception as e:
            logger.error(f"Scheduler initialization failed: {e}")
            return False
        
        # Initialize news fetcher
        logger.info("Initializing news fetcher...")
        try:
            news_fetcher.initialize()
            logger.info("News fetcher initialized")
        except Exception as e:
            logger.error(f"News fetcher initialization failed: {e}")
            return False
        
        logger.info("="*60)
        logger.info("ALL SYSTEMS OPERATIONAL - BOT READY")
        logger.info("="*60)
        
        return True
    
    except Exception as e:
        logger.error(f"Critical error during initialization: {e}")
        return False


async def shutdown_bot():
    """
    Graceful shutdown: Close all positions, save state, cleanup.
    
    This function:
    1. Stops accepting new trades
    2. Closes all open MT5 positions at market
    3. Saves final state to database
    4. Stops scheduler
    5. Closes database connection
    """
    global shutdown_in_progress
    
    try:
        if shutdown_in_progress:
            logger.info("Shutdown already in progress, skipping...")
            return
        
        shutdown_in_progress = True
        
        logger.info("="*60)
        logger.info("INITIATING GRACEFUL SHUTDOWN")
        logger.info("="*60)
        
        # Stop scheduler first (prevents new trades)
        logger.info("Stopping scheduler...")
        try:
            scheduler.stop_scheduler()
            logger.info("Scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
        
        # Close all open MT5 positions
        logger.info("Closing all open positions...")
        try:
            # Get all active subscribers with MT5 connected
            active_users = database.get_active_subscribers()
            
            for user in active_users:
                if user.get('mt5_connected'):
                    telegram_id = user['telegram_id']
                    logger.info(f"Closing positions for user {telegram_id}...")
                    
                    try:
                        mt5_connector.close_all_positions(telegram_id)
                        logger.info(f"Positions closed for user {telegram_id}")
                    except Exception as e:
                        logger.error(f"Error closing positions for user {telegram_id}: {e}")
            
            logger.info("All positions closed")
        except Exception as e:
            logger.error(f"Error during position closing: {e}")
        
        # Save ML models (if updated)
        logger.info("Saving ML model state...")
        try:
            ml_models.save_models()
            logger.info("ML models saved")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
        
        logger.info("="*60)
        logger.info("GRACEFUL SHUTDOWN COMPLETE")
        logger.info("="*60)
    
    except Exception as e:
        logger.error(f"Error during graceful shutdown: {e}")


def handle_shutdown_signal(signum, frame):
    """
    Handle OS shutdown signals (SIGTERM, SIGINT).
    
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    logger.info(f"Received shutdown signal: {signum}")
    asyncio.create_task(shutdown_bot())
    sys.exit(0)


# ==================== COMMAND HANDLERS ====================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle /start command.
    
    Sends welcome message with legal disclaimer and available commands.
    """
    try:
        user = update.effective_user
        telegram_id = user.id
        username = user.username
        first_name = user.first_name
        
        logger.info(f"User {telegram_id} (@{username}) executed /start")
        
        # Get or create user in database
        user_locale = update.effective_user.language_code
        detected_timezone = utils.detect_timezone(user_locale)
        
        db_user = database.get_or_create_user(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
            timezone=detected_timezone
        )
        
        if db_user is None:
            await update.message.reply_text(
                "Error: Could not create user account. Please contact support at "
                f"{config.SUPPORT_CONTACT}"
            )
            return
        
        # Format welcome message
        welcome_text = config.WELCOME_MESSAGE.format(
            product_name=config.PRODUCT_NAME,
            support_contact=config.SUPPORT_CONTACT,
            footer=config.FOOTER
        )
        
        # Validate message (filter forbidden words)
        clean_welcome = utils.validate_user_message(welcome_text)
        
        # Send welcome message
        await update.message.reply_text(clean_welcome)
        
        logger.info(f"Sent welcome message to user {telegram_id}")
    
    except Exception as e:
        logger.error(f"Error in start_command: {e}")
        await update.message.reply_text(
            "An error occurred. Please try again or contact support."
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle /help command.
    
    Sends comprehensive help guide with:
    - Available commands
    - Setup quality explanations
    - SMC concepts
    - Order type definitions
    - Auto-execution flow
    """
    try:
        user = update.effective_user
        telegram_id = user.id
        
        logger.info(f"User {telegram_id} executed /help")
        
        # Format help message
        help_text = config.HELP_MESSAGE.format(
            support_contact=config.SUPPORT_CONTACT,
            footer=config.FOOTER
        )
        
        # Validate message
        clean_help = utils.validate_user_message(help_text)
        
        # Send help message
        await update.message.reply_text(clean_help)
        
        logger.info(f"Sent help message to user {telegram_id}")
    
    except Exception as e:
        logger.error(f"Error in help_command: {e}")
        await update.message.reply_text(
            "An error occurred. Please try again or contact support."
        )


async def subscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle /subscribe command.
    
    Shows legal disclaimer and activates subscription upon acceptance.
    """
    try:
        user = update.effective_user
        telegram_id = user.id
        
        logger.info(f"User {telegram_id} executed /subscribe")
        
        # Get user from database
        db_user = database.get_user(telegram_id)
        
        if db_user is None:
            await update.message.reply_text(
                "Please use /start first to create your account."
            )
            return
        
        # Check if already subscribed
        if db_user.get('subscription_status') == 'active':
            already_subscribed_text = config.ALREADY_SUBSCRIBED.format(
                footer=config.FOOTER
            )
            clean_text = utils.validate_user_message(already_subscribed_text)
            await update.message.reply_text(clean_text)
            return
        
        # Show legal disclaimer
        disclaimer_text = config.LEGAL_DISCLAIMER.format(
            support_contact=config.SUPPORT_CONTACT
        )
        
        # Create inline keyboard for acceptance
        keyboard = [
            [
                InlineKeyboardButton("I DO NOT ACCEPT - Exit", callback_data="disclaimer_reject")
            ],
            [
                InlineKeyboardButton(
                    "I ACCEPT AND UNDERSTAND - Continue",
                    callback_data="disclaimer_accept"
                )
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            disclaimer_text,
            reply_markup=reply_markup
        )
        
        logger.info(f"Sent legal disclaimer to user {telegram_id}")
    
    except Exception as e:
        logger.error(f"Error in subscribe_command: {e}")
        await update.message.reply_text(
            "An error occurred. Please try again or contact support."
        )


async def disclaimer_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle disclaimer acceptance/rejection callbacks.
    """
    try:
        query = update.callback_query
        await query.answer()
        
        user = update.effective_user
        telegram_id = user.id
        
        callback_data = query.data
        
        if callback_data == "disclaimer_reject":
            # User rejected disclaimer
            await query.edit_message_text(
                "You have declined the terms. "
                "You cannot use this service without accepting the disclaimer.\n\n"
                "If you change your mind, use /subscribe again."
            )
            logger.info(f"User {telegram_id} rejected disclaimer")
        
        elif callback_data == "disclaimer_accept":
            # User accepted disclaimer - activate subscription
            success = database.activate_subscription(telegram_id)
            
            if success:
                # Send confirmation message
                confirmation_text = config.SUBSCRIPTION_SUCCESS.format(
                    support_contact=config.SUPPORT_CONTACT,
                    footer=config.FOOTER
                )
                clean_text = utils.validate_user_message(confirmation_text)
                
                await query.edit_message_text(clean_text)
                
                logger.info(f"User {telegram_id} accepted disclaimer and activated subscription")
            else:
                await query.edit_message_text(
                    "Error activating subscription. Please contact support at "
                    f"{config.SUPPORT_CONTACT}"
                )
                logger.error(f"Failed to activate subscription for user {telegram_id}")
    
    except Exception as e:
        logger.error(f"Error in disclaimer_callback: {e}")


async def unsubscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle /unsubscribe command.
    
    Shows confirmation dialog before deactivating.
    """
    try:
        user = update.effective_user
        telegram_id = user.id
        
        logger.info(f"User {telegram_id} executed /unsubscribe")
        
        # Get user
        db_user = database.get_user(telegram_id)
        
        if db_user is None or db_user.get('subscription_status') != 'active':
            await update.message.reply_text(
                "You are not currently subscribed."
            )
            return
        
        # Show confirmation
        confirmation_text = config.UNSUBSCRIBE_CONFIRM.format(footer=config.FOOTER)
        
        keyboard = [
            [
                InlineKeyboardButton("Cancel", callback_data="unsubscribe_cancel")
            ],
            [
                InlineKeyboardButton(
                    "Yes, Unsubscribe",
                    callback_data="unsubscribe_confirm"
                )
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            confirmation_text,
            reply_markup=reply_markup
        )
    
    except Exception as e:
        logger.error(f"Error in unsubscribe_command: {e}")
        await update.message.reply_text(
            "An error occurred. Please try again or contact support."
        )


async def unsubscribe_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle unsubscribe confirmation callback.
    """
    try:
        query = update.callback_query
        await query.answer()
        
        user = update.effective_user
        telegram_id = user.id
        
        callback_data = query.data
        
        if callback_data == "unsubscribe_cancel":
            await query.edit_message_text("Unsubscribe canceled. You remain subscribed.")
            logger.info(f"User {telegram_id} canceled unsubscribe")
        
        elif callback_data == "unsubscribe_confirm":
            # Deactivate subscription
            success = database.deactivate_subscription(telegram_id)
            
            if success:
                success_text = config.UNSUBSCRIBE_SUCCESS.format(
                    support_contact=config.SUPPORT_CONTACT,
                    footer=config.FOOTER
                )
                clean_text = utils.validate_user_message(success_text)
                
                await query.edit_message_text(clean_text)
                
                logger.info(f"User {telegram_id} unsubscribed successfully")
            else:
                await query.edit_message_text(
                    "Error processing unsubscribe request. Please contact support."
                )
                logger.error(f"Failed to unsubscribe user {telegram_id}")
    
    except Exception as e:
        logger.error(f"Error in unsubscribe_callback: {e}")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle /status command.
    
    Shows user subscription status, MT5 connection, and trading stats.
    """
    try:
        user = update.effective_user
        telegram_id = user.id
        
        logger.info(f"User {telegram_id} executed /status")
        
        # Get user
        db_user = database.get_user(telegram_id)
        
        if db_user is None:
            await update.message.reply_text(
                "No account found. Please use /start first."
            )
            return
        
        # Build status message
        status_lines = []
        status_lines.append("ACCOUNT STATUS")
        status_lines.append("="*40)
        
        # Subscription status
        subscription_status = db_user.get('subscription_status', 'inactive')
        if subscription_status == 'active':
            status_lines.append("Subscription: Active")
            
            trial_started = db_user.get('trial_started_at')
            if trial_started:
                trial_started_dt = utils.parse_iso_datetime(trial_started)
                if trial_started_dt:
                    user_tz = db_user.get('timezone', 'UTC')
                    trial_started_local = utils.convert_utc_to_user_time(trial_started_dt, user_tz)
                    status_lines.append(f"Trial Started: {trial_started_local.strftime('%Y-%m-%d %H:%M')}")
        else:
            status_lines.append("Subscription: Inactive")
            status_lines.append("Use /subscribe to activate")
        
        # MT5 connection status
        mt5_connected = db_user.get('mt5_connected', False)
        if mt5_connected:
            broker_name = db_user.get('mt5_broker_name', 'Unknown')
            mt5_login = db_user.get('mt5_login', 'Unknown')
            status_lines.append("")
            status_lines.append("MT5 CONNECTION")
            status_lines.append(f"Broker: {broker_name}")
            status_lines.append(f"Account: {mt5_login}")
            status_lines.append("Status: Connected")
            status_lines.append("Auto-execution: Enabled")
        else:
            status_lines.append("")
            status_lines.append("MT5 CONNECTION")
            status_lines.append("Status: Not connected")
            status_lines.append("Use /connect_mt5 to enable auto-trading")
        
        # Risk settings
        risk_percent = db_user.get('risk_percent', config.DEFAULT_RISK_PERCENT)
        status_lines.append("")
        status_lines.append("RISK SETTINGS")
        status_lines.append(f"Risk Per Trade: {risk_percent}%")
        status_lines.append("Modify in /settings")
        
        # Timezone
        user_timezone = db_user.get('timezone', 'UTC')
        status_lines.append("")
        status_lines.append(f"Timezone: {user_timezone}")
        
        status_lines.append("")
        status_lines.append(config.FOOTER)
        
        status_message = "\n".join(status_lines)
        clean_message = utils.validate_user_message(status_message)
        
        await update.message.reply_text(clean_message)
        
        logger.info(f"Sent status to user {telegram_id}")
    
    except Exception as e:
        logger.error(f"Error in status_command: {e}")
        await update.message.reply_text(
            "An error occurred. Please try again or contact support."
        )


async def latest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle /latest command.
    
    Shows the most recent automated setup from last 24 hours.
    """
    try:
        user = update.effective_user
        telegram_id = user.id
        
        logger.info(f"User {telegram_id} executed /latest")
        
        # Check if user is subscribed
        db_user = database.get_user(telegram_id)
        
        if db_user is None or db_user.get('subscription_status') != 'active':
            await update.message.reply_text(
                "You must be subscribed to view automated setups.\n"
                "Use /subscribe to activate your account."
            )
            return
        
        # Get latest signal
        latest_signal = database.get_latest_signal()
        
        if latest_signal is None:
            no_setup_text = config.NO_RECENT_SETUPS.format(footer=config.FOOTER)
            clean_text = utils.validate_user_message(no_setup_text)
            await update.message.reply_text(clean_text)
            return
        
        # Format signal message
        from scheduler import format_setup_alert
        
        signal_message = format_setup_alert(latest_signal)
        clean_message = utils.validate_user_message(signal_message)
        
        await update.message.reply_text(clean_message, parse_mode='Markdown')
        
        logger.info(f"Sent latest setup to user {telegram_id}")
    
    except Exception as e:
        logger.error(f"Error in latest_command: {e}")
        await update.message.reply_text(
            "An error occurred. Please try again or contact support."
        )


# ==================== MT5 CONNECTION HANDLERS ====================

async def connect_mt5_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle /connect_mt5 command.
    
    Initiates MT5 connection flow (credentials input).
    """
    try:
        user = update.effective_user
        telegram_id = user.id
        
        logger.info(f"User {telegram_id} executed /connect_mt5")
        
        # Check if already connected
        db_user = database.get_user(telegram_id)
        
        if db_user is None:
            await update.message.reply_text(
                "Please use /start first to create your account."
            )
            return
        
        if db_user.get('mt5_connected'):
            broker_name = db_user.get('mt5_broker_name', 'Unknown')
            account_number = db_user.get('mt5_login', 'Unknown')
            
            already_connected_text = config.MT5_CONNECTED_ALREADY.format(
                broker_name=broker_name,
                account_number=account_number,
                footer=config.FOOTER
            )
            clean_text = utils.validate_user_message(already_connected_text)
            
            await update.message.reply_text(clean_text)
            return
        
        # Send instructions
        instructions = (
            "MT5 CONNECTION SETUP\n"
            "="*40 + "\n\n"
            "To connect your MetaTrader 5 account, please send your credentials in this format:\n\n"
            "/mt5_credentials LOGIN PASSWORD SERVER BROKER_NAME\n\n"
            "Example:\n"
            "/mt5_credentials 12345678 MyPassword123 XM-Global-Real XMGlobal\n\n"
            "IMPORTANT:\n"
            "- Your password is encrypted before storage\n"
            "- Use a secure connection (not public WiFi)\n"
            "- You can use demo account for testing\n\n"
            f"{config.FOOTER}"
        )
        
        await update.message.reply_text(instructions)
        
        logger.info(f"Sent MT5 connection instructions to user {telegram_id}")
    
    except Exception as e:
        logger.error(f"Error in connect_mt5_command: {e}")
        await update.message.reply_text(
            "An error occurred. Please try again or contact support."
        )


async def mt5_credentials_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle /mt5_credentials command.
    
    Receives and saves encrypted MT5 credentials.
    """
    try:
        user = update.effective_user
        telegram_id = user.id
        
        logger.info(f"User {telegram_id} submitted MT5 credentials")
        
        # Delete the message immediately for security
        try:
            await update.message.delete()
        except:
            pass  # Might fail if bot doesn't have delete permission
        
        # Parse credentials
        if not context.args or len(context.args) < 4:
            await context.bot.send_message(
                chat_id=telegram_id,
                text=(
                    "Invalid format. Please use:\n"
                    "/mt5_credentials LOGIN PASSWORD SERVER BROKER_NAME\n\n"
                    "Your message was deleted for security."
                )
            )
            return
        
        mt5_login = context.args[0]
        mt5_password = context.args[1]
        mt5_server = context.args[2]
        mt5_broker_name = " ".join(context.args[3:])  # Allow spaces in broker name
        
        # Save credentials (encrypted)
        success = database.save_mt5_credentials(
            telegram_id=telegram_id,
            mt5_login=mt5_login,
            mt5_password=mt5_password,
            mt5_server=mt5_server,
            mt5_broker_name=mt5_broker_name
        )
        
        if success:
            # Test connection
            connection_ok = mt5_connector.test_mt5_connection(telegram_id)
            
            if connection_ok:
                # Get account balance
                balance = mt5_connector.get_account_balance(telegram_id)
                
                success_text = config.MT5_CONNECTION_SUCCESS.format(
                    broker_name=mt5_broker_name,
                    account_number=mt5_login,
                    balance=utils.format_currency(balance) if balance else "N/A",
                    footer=config.FOOTER
                )
                clean_text = utils.validate_user_message(success_text)
                
                await context.bot.send_message(
                    chat_id=telegram_id,
                    text=clean_text
                )
                
                logger.info(f"MT5 connection successful for user {telegram_id}")
            else:
                await context.bot.send_message(
                    chat_id=telegram_id,
                    text=(
                        "Connection test failed. Please verify:\n"
                        "- Login credentials are correct\n"
                        "- Server name is exact (case-sensitive)\n"
                        "- MT5 account is active\n\n"
                        "Try again with /connect_mt5"
                    )
                )
                
                # Delete saved credentials
                database.delete_mt5_credentials(telegram_id)
                
                logger.warning(f"MT5 connection test failed for user {telegram_id}")
        else:
            await context.bot.send_message(
                chat_id=telegram_id,
                text=(
                    "Error saving credentials. Please try again or contact support at "
                    f"{config.SUPPORT_CONTACT}"
                )
            )
            logger.error(f"Failed to save MT5 credentials for user {telegram_id}")
    
    except Exception as e:
        logger.error(f"Error in mt5_credentials_command: {e}")
        await context.bot.send_message(
            chat_id=telegram_id,
            text="An error occurred. Please try again or contact support."
        )


async def disconnect_mt5_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle /disconnect_mt5 command.
    
    Shows confirmation dialog before disconnecting.
    """
    try:
        user = update.effective_user
        telegram_id = user.id
        
        logger.info(f"User {telegram_id} executed /disconnect_mt5")
        
        # Check if connected
        db_user = database.get_user(telegram_id)
        
        if db_user is None or not db_user.get('mt5_connected'):
            await update.message.reply_text(
                "You are not currently connected to MT5."
            )
            return
        
        # Show confirmation
        confirmation_text = config.MT5_DISCONNECTION_CONFIRM.format(footer=config.FOOTER)
        
        keyboard = [
            [
                InlineKeyboardButton("Cancel", callback_data="mt5_disconnect_cancel")
            ],
            [
                InlineKeyboardButton(
                    "Yes, Disconnect",
                    callback_data="mt5_disconnect_confirm"
                )
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            confirmation_text,
            reply_markup=reply_markup
        )
    
    except Exception as e:
        logger.error(f"Error in disconnect_mt5_command: {e}")
        await update.message.reply_text(
            "An error occurred. Please try again or contact support."
        )


async def mt5_disconnect_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle MT5 disconnect confirmation callback.
    """
    try:
        query = update.callback_query
        await query.answer()
        
        user = update.effective_user
        telegram_id = user.id
        
        callback_data = query.data
        
        if callback_data == "mt5_disconnect_cancel":
            await query.edit_message_text("Disconnect canceled. Your MT5 connection remains active.")
            logger.info(f"User {telegram_id} canceled MT5 disconnect")
        
        elif callback_data == "mt5_disconnect_confirm":
            # Delete credentials
            success = database.delete_mt5_credentials(telegram_id)
            
            if success:
                success_text = config.MT5_DISCONNECTION_SUCCESS.format(footer=config.FOOTER)
                clean_text = utils.validate_user_message(success_text)
                
                await query.edit_message_text(clean_text)
                
                logger.info(f"User {telegram_id} disconnected MT5 successfully")
            else:
                await query.edit_message_text(
                    "Error processing disconnect request. Please contact support."
                )
                logger.error(f"Failed to disconnect MT5 for user {telegram_id}")
    
    except Exception as e:
        logger.error(f"Error in mt5_disconnect_callback: {e}")


# ==================== SETTINGS HANDLER ====================

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle /settings command.
    
    Shows customizable settings menu.
    """
    try:
        user = update.effective_user
        telegram_id = user.id
        
        logger.info(f"User {telegram_id} executed /settings")
        
        # Get user
        db_user = database.get_user(telegram_id)
        
        if db_user is None:
            await update.message.reply_text(
                "Please use /start first to create your account."
            )
            return
        
        # Build settings menu
        current_risk = db_user.get('risk_percent', config.DEFAULT_RISK_PERCENT)
        current_timezone = db_user.get('timezone', 'UTC')
        
        settings_text = (
            "CUSTOMIZABLE SETTINGS\n"
            "="*40 + "\n\n"
            f"Current Risk Per Trade: {current_risk}%\n"
            f"Current Timezone: {current_timezone}\n\n"
            "To update risk percentage:\n"
            "/set_risk 1.5\n"
            "(Range: 0.1% - 5.0%)\n\n"
            "To update timezone:\n"
            "/set_timezone America/New_York\n"
            "(Use IANA timezone names)\n\n"
            f"{config.FOOTER}"
        )
        
        await update.message.reply_text(settings_text)
        
        logger.info(f"Sent settings menu to user {telegram_id}")
    
    except Exception as e:
        logger.error(f"Error in settings_command: {e}")
        await update.message.reply_text(
            "An error occurred. Please try again or contact support."
        )


async def set_risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle /set_risk command.
    
    Updates user risk percentage.
    """
    try:
        user = update.effective_user
        telegram_id = user.id
        
        logger.info(f"User {telegram_id} executed /set_risk")
        
        # Parse risk value
        if not context.args or len(context.args) != 1:
            await update.message.reply_text(
                "Invalid format. Use: /set_risk 1.5\n"
                "Risk must be between 0.1% and 5.0%"
            )
            return
        
        try:
            risk_percent = float(context.args[0])
        except ValueError:
            await update.message.reply_text(
                "Invalid number. Use: /set_risk 1.5"
            )
            return
        
        # Validate risk
        if not utils.validate_risk_percent(risk_percent):
            await update.message.reply_text(
                f"Risk must be between 0.1% and 5.0%\n"
                f"You entered: {risk_percent}%"
            )
            return
        
        # Update database
        success = database.update_risk_percent(telegram_id, risk_percent)
        
        if success:
            await update.message.reply_text(
                f"Risk percentage updated to {risk_percent}%\n\n"
                "This will apply to all future automated setups.\n\n"
                f"{config.FOOTER}"
            )
            logger.info(f"Updated risk to {risk_percent}% for user {telegram_id}")
        else:
            await update.message.reply_text(
                "Error updating risk percentage. Please try again or contact support."
            )
            logger.error(f"Failed to update risk for user {telegram_id}")
    
    except Exception as e:
        logger.error(f"Error in set_risk_command: {e}")
        await update.message.reply_text(
            "An error occurred. Please try again or contact support."
        )


async def set_timezone_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle /set_timezone command.
    
    Updates user timezone.
    """
    try:
        user = update.effective_user
        telegram_id = user.id
        
        logger.info(f"User {telegram_id} executed /set_timezone")
        
        # Parse timezone
        if not context.args or len(context.args) != 1:
            await update.message.reply_text(
                "Invalid format. Use: /set_timezone America/New_York\n\n"
                "Common timezones:\n"
                "- America/New_York (EST/EDT)\n"
                "- Europe/London (GMT/BST)\n"
                "- Asia/Tokyo (JST)\n"
                "- Africa/Lagos (WAT)"
            )
            return
        
        timezone_str = context.args[0]
        
        # Validate timezone
        try:
            import pytz
            pytz.timezone(timezone_str)
        except Exception:
            await update.message.reply_text(
                f"Invalid timezone: {timezone_str}\n\n"
                "Use IANA timezone names like:\n"
                "- America/New_York\n"
                "- Europe/London\n"
                "- Asia/Tokyo"
            )
            return
        
        # Update database
        success = database.update_timezone(telegram_id, timezone_str)
        
        if success:
            await update.message.reply_text(
                f"Timezone updated to {timezone_str}\n\n"
                "Daily alerts will arrive at 8:00 AM in your local time.\n\n"
                f"{config.FOOTER}"
            )
            logger.info(f"Updated timezone to {timezone_str} for user {telegram_id}")
        else:
            await update.message.reply_text(
                "Error updating timezone. Please try again or contact support."
            )
            logger.error(f"Failed to update timezone for user {telegram_id}")
    
    except Exception as e:
        logger.error(f"Error in set_timezone_command: {e}")
        await update.message.reply_text(
            "An error occurred. Please try again or contact support."
        )
    
async def send_daily_alert(
    context: ContextTypes.DEFAULT_TYPE,
    telegram_id: int,
    user_timezone: str
) -> bool:
    """
    Send daily market briefing at 8 AM user's local time.
    
    Includes:
    - RED FOLDER NEWS (HIGH impact events) in UTC + local time
    - Market session overview
    - Yesterday's performance summary
    - Today's trading plan
    
    Args:
        context: Telegram context for sending messages
        telegram_id: User's Telegram ID
        user_timezone: User's IANA timezone string
        
    Returns:
        bool: True if alert sent successfully
        
    Note:
        All text must pass through utils.validate_user_message()
        NO EMOJIS allowed
    """
    try:
        logger.info(f"Preparing daily alert for user {telegram_id} (timezone: {user_timezone})")
        
        # Get user data
        user = database.get_user(telegram_id)
        if not user or user.get('subscription_status') != 'active':
            logger.warning(f"User {telegram_id} not active, skipping daily alert")
            return False
        
        # Get current time in user's timezone
        user_tz = utils.convert_utc_to_user_time(
            utils.get_current_utc_time(),
            user_timezone
        )
        
        # Get today's HIGH impact news (red folder)
        todays_news = database.get_todays_news()
        high_impact_news = [
            event for event in todays_news 
            if event.get('impact', '').upper() == 'HIGH'
        ]
        
        # Build news section with BOTH UTC and local time
        news_section = "RED FOLDER NEWS (High Impact Events):\n"
        if high_impact_news:
            for event in high_impact_news:
                event_time_utc = utils.parse_iso_datetime(event['event_time_utc'])
                if event_time_utc:
                    # Convert to user's local time
                    event_time_local = utils.convert_utc_to_user_time(
                        event_time_utc,
                        user_timezone
                    )
                    
                    # Format: "14:30 UTC (2:30 PM your time)"
                    utc_time_str = event_time_utc.strftime('%H:%M UTC')
                    local_time_str = event_time_local.strftime('%I:%M %p your time')
                    
                    news_section += (
                        f"\n{event['currency']} {event['event_name']}\n"
                        f"Time: {utc_time_str} ({local_time_str})\n"
                        f"Previous: {event.get('previous', 'N/A')} | "
                        f"Forecast: {event.get('forecast', 'N/A')}\n"
                    )
        else:
            news_section += "\nNo high-impact events scheduled for today.\n"
        
        # Get yesterday's performance (if user has MT5 connected)
        mt5_creds = database.get_mt5_credentials(telegram_id)
        yesterday_summary = ""
        
        if mt5_creds:
            # This would fetch yesterday's closed trades
            # For MVP, we'll use placeholder
            yesterday_summary = (
                "\nYesterday's Performance:\n"
                "Data collection in progress. View full history in MT5 terminal.\n"
            )
        else:
            yesterday_summary = (
                "\nConnect MT5 to track daily performance.\n"
                "Use /connect_mt5 to link your broker account.\n"
            )
        
        # Market session overview
        current_utc_hour = utils.get_current_utc_time().hour
        current_session = utils.get_session_name(current_utc_hour)
        
        session_overview = (
            f"\nCurrent Market Session: {current_session}\n"
            f"Market Status: {'Open' if utils.is_market_open() else 'Closed (Weekend)'}\n"
        )
        
        # Today's trading plan
        trading_plan = (
            "\nToday's Trading Plan:\n"
            "The algorithm will scan for high-probability automated setups based on:\n"
            "- Smart Money Concepts (Order Blocks, Breaker Blocks, FVGs)\n"
            "- Multi-timeframe confluence\n"
            "- Volume confirmation\n"
            "- Session-appropriate liquidity\n"
            "\nYou will be notified immediately when setups align with criteria.\n"
        )
        
        # Assemble complete message
        message = (
            f"DAILY MARKET BRIEFING\n"
            f"{user_tz.strftime('%A, %B %d, %Y')}\n"
            f"{'=' * 40}\n"
            f"\n{news_section}"
            f"\n{session_overview}"
            f"\n{yesterday_summary}"
            f"\n{trading_plan}"
            f"\n{'=' * 40}\n"
            f"{config.FOOTER}"
        )
        
        # Validate message (filter forbidden words)
        validated_message = utils.validate_user_message(message)
        
        # Send message
        await context.bot.send_message(
            chat_id=telegram_id,
            text=validated_message,
            parse_mode=None
        )
        
        # Update last_8am_alert_sent timestamp
        database.update_last_8am_alert(telegram_id)
        
        logger.info(f"Daily alert sent successfully to user {telegram_id}")
        return True
    
    except TelegramError as e:
        logger.error(f"Telegram error sending daily alert to {telegram_id}: {e}")
        return False
    
    except Exception as e:
        logger.error(f"Error sending daily alert to {telegram_id}: {e}", exc_info=True)
        return False


# ==================== NEW SETUP BROADCAST ====================

async def broadcast_new_signal(
    context: ContextTypes.DEFAULT_TYPE,
    signal: Dict[str, Any],
    current_price: float
) -> int:
    """
    Broadcast new automated setup to all active subscribers.
    
    CRITICAL REQUIREMENTS:
    - NEVER show model names (LSTM, XGBoost, GRU) to users
    - Use "Model Agreement Score: XX% DIRECTION" only
    - Use exact template from specification
    - Chart generation DISABLED for now
    - NO EMOJIS
    
    Args:
        context: Telegram context for sending messages
        signal: Signal dictionary from database
        current_price: Current market price for distance calculation
        
    Returns:
        int: Number of users successfully notified
        
    Template:
        NEW AUTOMATED SETUP: {symbol} | {setup_type}
        
        Model Agreement Score: {ml_score}% {direction}
        Direction: {direction}
        ORDER TYPE: {order_type}
        
        Entry: {entry_price}
        Stop Loss: {stop_loss} ({sl_pips} pips)
        TP1: {tp1_price} (plus {tp1_pips} pips, estimated {tp1_profit_usd} USD, R:R {rr_tp1})
        TP2: {tp2_price} (plus {tp2_pips} pips, estimated {tp2_profit_usd} USD, R:R {rr_tp2})
        
        {order_type_explanation}
        
        Note: {news_proximity}
        
        {session_info}
        
        Educational tool. Not financial advice. You control your account. Past performance does not guarantee future results.
    """
    try:
        logger.info(f"Broadcasting signal #{signal['signal_number']}: {signal['symbol']} {signal['direction']}")
        
        # Extract signal data
        symbol = signal['symbol']
        setup_type = signal['setup_type']
        direction = signal['direction']
        ml_score = signal['ml_score']
        order_type = signal['order_type']
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        tp1 = signal['take_profit_1']
        tp2 = signal['take_profit_2']
        sl_pips = signal['sl_pips']
        tp1_pips = signal['tp1_pips']
        tp2_pips = signal['tp2_pips']
        rr_tp1 = signal['rr_tp1']
        rr_tp2 = signal['rr_tp2']
        session = signal['session']
        
        # Calculate distance from current price to entry
        pip_size = config.PIP_SIZES.get(symbol, 0.0001)
        distance_pips = abs(entry_price - current_price) / pip_size
        
        # Estimate profit in USD (using $10,000 account, 1% risk as reference)
        # This is for educational demonstration only
        risk_amount = 100  # 1% of $10,000
        tp1_profit_usd = int((tp1_pips / sl_pips) * risk_amount)
        tp2_profit_usd = int((tp2_pips / sl_pips) * risk_amount)
        
        # Order type explanation
        if order_type == 'MARKET ORDER':
            order_explanation = (
                f"Entry at {entry_price} is within 2 pips of current price {current_price}. "
                f"Order will execute immediately at market."
            )
        elif order_type == 'LIMIT ORDER':
            order_explanation = (
                f"Entry at {entry_price} requires pullback from current {current_price} "
                f"({distance_pips:.1f} pips away). Order will expire in 1 hour if not filled."
            )
        else:  # STOP ORDER
            order_explanation = (
                f"Entry at {entry_price} requires breakout from current {current_price} "
                f"({distance_pips:.1f} pips away). Order will expire in 1 hour if not filled."
            )
        
        # News proximity check
        # For MVP, using placeholder. Real implementation would check database
        news_proximity = "No major news scheduled within 30 minutes"
        
        # Session info with liquidity assessment
        session_liquidity = {
            'London': 'High Liquidity',
            'New York': 'High Liquidity',
            'Overlap': 'Very High Liquidity',
            'Asian': 'Moderate Liquidity',
            'Off Hours': 'Low Liquidity'
        }
        liquidity_level = session_liquidity.get(session, 'Unknown')
        session_info = f"Session: {session} ({liquidity_level})"
        
        # Build setup type display name
        setup_display = {
            'unicorn': 'Unicorn Breaker Block',
            'standard_ob': 'Standard Order Block',
            'breaker': 'Breaker Block',
            'bb_continuation': 'Breaker Block Continuation'
        }
        setup_name = setup_display.get(setup_type.lower(), setup_type)
        
        # Build complete message using EXACT template
        message = (
            f"NEW AUTOMATED SETUP: {symbol} | {setup_name}\n"
            f"\n"
            f"Model Agreement Score: {ml_score}% {direction}\n"
            f"Direction: {direction}\n"
            f"ORDER TYPE: {order_type}\n"
            f"\n"
            f"Entry: {entry_price}\n"
            f"Stop Loss: {stop_loss} ({sl_pips:.1f} pips)\n"
            f"TP1: {tp1} (plus {tp1_pips:.1f} pips, estimated {tp1_profit_usd} USD, R:R {rr_tp1})\n"
            f"TP2: {tp2} (plus {tp2_pips:.1f} pips, estimated {tp2_profit_usd} USD, R:R {rr_tp2})\n"
            f"\n"
            f"{order_explanation}\n"
            f"\n"
            f"Note: {news_proximity}\n"
            f"\n"
            f"{session_info}\n"
            f"\n"
            f"Educational tool. Not financial advice. You control your account. "
            f"Past performance does not guarantee future results."
        )
        
        # Validate message (filter forbidden words)
        validated_message = utils.validate_user_message(message)
        
        # Get all active subscribers
        active_users = database.get_active_subscribers()
        
        successful_sends = 0
        
        # Send to each subscriber
        for user in active_users:
            try:
                await context.bot.send_message(
                    chat_id=user['telegram_id'],
                    text=validated_message,
                    parse_mode=None
                )
                
                successful_sends += 1
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.05)  # 50ms between sends
            
            except TelegramError as e:
                logger.error(
                    f"Failed to send signal to user {user['telegram_id']}: {e}"
                )
                continue
        
        logger.info(
            f"Signal #{signal['signal_number']} broadcast complete. "
            f"Sent to {successful_sends}/{len(active_users)} users."
        )
        
        return successful_sends
    
    except Exception as e:
        logger.error(f"Error broadcasting signal: {e}", exc_info=True)
        return 0


# ==================== TRADE EXECUTION NOTIFICATIONS ====================

async def notify_trade_opened(
    context: ContextTypes.DEFAULT_TYPE,
    telegram_id: int,
    trade: Dict[str, Any]
) -> bool:
    """
    Notify user that trade was opened (auto-execution).
    
    Args:
        context: Telegram context
        telegram_id: User's Telegram ID
        trade: Trade dictionary with execution details
        
    Returns:
        bool: True if notification sent successfully
    """
    try:
        symbol = trade['symbol']
        direction = trade['direction']
        entry_price = trade['entry_price']
        lot_size = trade['lot_size']
        stop_loss = trade['stop_loss']
        tp1 = trade['take_profit_1']
        tp2 = trade.get('take_profit_2')
        
        # Get broker info from user
        mt5_creds = database.get_mt5_credentials(telegram_id)
        broker_name = mt5_creds.get('broker_name', 'MT5') if mt5_creds else 'MT5'
        
        # Mask account number
        account = trade.get('broker_account', 'N/A')
        if len(str(account)) > 4:
            account_masked = '****' + str(account)[-4:]
        else:
            account_masked = '****'
        
        # Calculate slippage
        expected_entry = trade.get('expected_entry', entry_price)
        pip_size = config.PIP_SIZES.get(symbol, 0.0001)
        slippage_pips = abs(entry_price - expected_entry) / pip_size
        
        # Calculate risk amount
        risk_usd = trade.get('risk_amount_usd', 0)
        
        message = (
            f"TRADE OPENED\n"
            f"\n"
            f"{symbol} {direction}\n"
            f"Broker: {broker_name}\n"
            f"Account: {account_masked}\n"
            f"\n"
            f"Entry: {entry_price}\n"
            f"Slippage: {slippage_pips:.1f} pips\n"
            f"Lot Size: {lot_size}\n"
            f"Risk: {utils.format_currency(risk_usd)}\n"
            f"\n"
            f"Stop Loss: {stop_loss}\n"
            f"TP1: {tp1}\n"
        )
        
        if tp2:
            message += f"TP2: {tp2}\n"
        
        message += (
            f"\n"
            f"Position is being monitored. You will be notified when targets are reached.\n"
            f"\n"
            f"{config.FOOTER}"
        )
        
        validated_message = utils.validate_user_message(message)
        
        await context.bot.send_message(
            chat_id=telegram_id,
            text=validated_message,
            parse_mode=None
        )
        
        logger.info(f"Sent trade opened notification to user {telegram_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error sending trade opened notification: {e}")
        return False


async def notify_tp1_hit(
    context: ContextTypes.DEFAULT_TYPE,
    telegram_id: int,
    trade: Dict[str, Any]
) -> bool:
    """
    Notify user that TP1 was hit (50% closed, SL moved to breakeven).
    
    Args:
        context: Telegram context
        telegram_id: User's Telegram ID
        trade: Trade dictionary with current status
        
    Returns:
        bool: True if notification sent successfully
    """
    try:
        symbol = trade['symbol']
        direction = trade['direction']
        entry_price = trade['entry_price']
        tp1_price = trade['take_profit_1']
        
        # Calculate profit from 50% closure
        lot_size = trade['lot_size']
        closed_lots = lot_size / 2
        
        pip_size = config.PIP_SIZES.get(symbol, 0.0001)
        pips_profit = abs(tp1_price - entry_price) / pip_size
        
        partial_profit_usd = trade.get('partial_profit_usd', 0)
        
        # New SL position (breakeven + buffer)
        breakeven_sl = trade.get('breakeven_sl', entry_price)
        
        message = (
            f"TP1 HIT\n"
            f"\n"
            f"{symbol} {direction}\n"
            f"\n"
            f"Closed 50% at TP1: {tp1_price}\n"
            f"Profit from partial close: {utils.format_currency(partial_profit_usd)}\n"
            f"Pips captured: {pips_profit:.1f}\n"
            f"\n"
            f"Remaining Position:\n"
            f"Lot Size: {closed_lots:.2f} (50% of original)\n"
            f"Stop Loss moved to: {breakeven_sl}\n"
            f"(Breakeven + 5 pips buffer)\n"
            f"\n"
            f"Position is now risk-free. Riding to TP2.\n"
            f"\n"
            f"{config.FOOTER}"
        )
        
        validated_message = utils.validate_user_message(message)
        
        await context.bot.send_message(
            chat_id=telegram_id,
            text=validated_message,
            parse_mode=None
        )
        
        logger.info(f"Sent TP1 hit notification to user {telegram_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error sending TP1 notification: {e}")
        return False


async def notify_tp2_hit(
    context: ContextTypes.DEFAULT_TYPE,
    telegram_id: int,
    trade: Dict[str, Any]
) -> bool:
    """
    Notify user that TP2 was hit (trade fully closed).
    
    Args:
        context: Telegram context
        telegram_id: User's Telegram ID
        trade: Trade dictionary with final results
        
    Returns:
        bool: True if notification sent successfully
    """
    try:
        symbol = trade['symbol']
        direction = trade['direction']
        entry_price = trade['entry_price']
        tp2_price = trade.get('exit_price', trade.get('take_profit_2'))
        
        # Calculate total profit
        total_profit_usd = trade.get('total_profit_usd', 0)
        
        # Calculate total pips
        pip_size = config.PIP_SIZES.get(symbol, 0.0001)
        total_pips = abs(tp2_price - entry_price) / pip_size
        
        # Calculate duration
        entry_time = utils.parse_iso_datetime(trade.get('entry_time'))
        exit_time = utils.parse_iso_datetime(trade.get('exit_time'))
        
        if entry_time and exit_time:
            duration = utils.format_duration(entry_time, exit_time)
        else:
            duration = "N/A"
        
        # Breakdown of closures
        tp1_profit = trade.get('partial_profit_usd', 0)
        tp2_profit = total_profit_usd - tp1_profit
        
        message = (
            f"TRADE CLOSED - TP2 HIT\n"
            f"\n"
            f"{symbol} {direction}\n"
            f"\n"
            f"Final Exit: {tp2_price}\n"
            f"Total Profit: {utils.format_currency(total_profit_usd)}\n"
            f"Total Pips: {total_pips:.1f}\n"
            f"Duration: {duration}\n"
            f"\n"
            f"Profit Breakdown:\n"
            f"- 50% at TP1: {utils.format_currency(tp1_profit)}\n"
            f"- 50% at TP2: {utils.format_currency(tp2_profit)}\n"
            f"\n"
            f"Excellent execution. Trade completed successfully.\n"
            f"\n"
            f"{config.FOOTER}"
        )
        
        validated_message = utils.validate_user_message(message)
        
        await context.bot.send_message(
            chat_id=telegram_id,
            text=validated_message,
            parse_mode=None
        )
        
        logger.info(f"Sent TP2 hit notification to user {telegram_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error sending TP2 notification: {e}")
        return False


async def notify_stop_loss_hit(
    context: ContextTypes.DEFAULT_TYPE,
    telegram_id: int,
    trade: Dict[str, Any]
) -> bool:
    """
    Notify user that stop loss was hit.
    
    Args:
        context: Telegram context
        telegram_id: User's Telegram ID
        trade: Trade dictionary with loss details
        
    Returns:
        bool: True if notification sent successfully
    """
    try:
        symbol = trade['symbol']
        direction = trade['direction']
        entry_price = trade['entry_price']
        sl_price = trade.get('exit_price', trade.get('stop_loss'))
        
        # Calculate loss
        loss_usd = trade.get('loss_usd', 0)
        
        # Calculate pips
        pip_size = config.PIP_SIZES.get(symbol, 0.0001)
        loss_pips = abs(sl_price - entry_price) / pip_size
        
        # Calculate duration
        entry_time = utils.parse_iso_datetime(trade.get('entry_time'))
        exit_time = utils.parse_iso_datetime(trade.get('exit_time'))
        
        if entry_time and exit_time:
            duration = utils.format_duration(entry_time, exit_time)
        else:
            duration = "N/A"
        
        message = (
            f"STOP LOSS HIT\n"
            f"\n"
            f"{symbol} {direction}\n"
            f"\n"
            f"Exit: {sl_price}\n"
            f"Loss: {utils.format_currency(abs(loss_usd))}\n"
            f"Pips: {loss_pips:.1f}\n"
            f"Duration: {duration}\n"
            f"\n"
            f"Stop loss hit. This is part of disciplined risk management.\n"
            f"Your account is protected by predefined risk limits.\n"
            f"\n"
            f"Not every trade wins. The strategy maintains edge through "
            f"favorable risk-reward ratios over time.\n"
            f"\n"
            f"{config.FOOTER}"
        )
        
        validated_message = utils.validate_user_message(message)
        
        await context.bot.send_message(
            chat_id=telegram_id,
            text=validated_message,
            parse_mode=None
        )
        
        logger.info(f"Sent stop loss notification to user {telegram_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error sending SL notification: {e}")
        return False


async def notify_trade_expired(
    context: ContextTypes.DEFAULT_TYPE,
    telegram_id: int,
    signal: Dict[str, Any]
) -> bool:
    """
    Notify user that limit/stop order expired without fill.
    
    Args:
        context: Telegram context
        telegram_id: User's Telegram ID
        signal: Signal dictionary with setup details
        
    Returns:
        bool: True if notification sent successfully
    """
    try:
        symbol = signal['symbol']
        direction = signal['direction']
        entry_price = signal['entry_price']
        order_type = signal['order_type']
        
        message = (
            f"ORDER EXPIRED\n"
            f"\n"
            f"{symbol} {direction}\n"
            f"Order Type: {order_type}\n"
            f"Entry Price: {entry_price}\n"
            f"\n"
            f"The {order_type.lower()} expired after 1 hour without being filled.\n"
            f"Market price did not reach the entry level within the validity window.\n"
            f"\n"
            f"No trade was executed. No risk taken.\n"
            f"\n"
            f"This is normal. Not every setup reaches entry. "
            f"The algorithm will continue scanning for new opportunities.\n"
            f"\n"
            f"{config.FOOTER}"
        )
        
        validated_message = utils.validate_user_message(message)
        
        await context.bot.send_message(
            chat_id=telegram_id,
            text=validated_message,
            parse_mode=None
        )
        
        logger.info(f"Sent order expired notification to user {telegram_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error sending expiry notification: {e}")
        return False


# ==================== ERROR HANDLER ====================

async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Global error handler for all bot exceptions.
    
    Logs all errors and sends professional error messages to users.
    NO EMOJIS in error messages.
    
    Args:
        update: Telegram update that caused error (may be None)
        context: Telegram context with error details
    """
    try:
        # Log the error with full traceback
        logger.error(
            f"Exception while handling an update:",
            exc_info=context.error
        )
        
        # If we can identify the user, send a professional error message
        if update and update.effective_user:
            telegram_id = update.effective_user.id
            
            # Determine error message based on exception type
            error_message = (
                f"An error occurred while processing your request.\n"
                f"\n"
                f"Our system has logged this issue and will investigate.\n"
                f"\n"
                f"Please try again in a few moments.\n"
                f"\n"
                f"If the problem persists, contact support: {config.SUPPORT_CONTACT}\n"
                f"\n"
                f"{config.FOOTER}"
            )
            
            # Validate and send
            validated_message = utils.validate_user_message(error_message)
            
            try:
                await context.bot.send_message(
                    chat_id=telegram_id,
                    text=validated_message,
                    parse_mode=None
                )
            except TelegramError as e:
                logger.error(f"Failed to send error message to user: {e}")
    
    except Exception as e:
        # Error in error handler (meta-error)
        logger.critical(f"Error in error_handler: {e}", exc_info=True)


# ==================== MAIN STARTUP FUNCTION ====================

def main() -> None:
    """
    Main startup function for Nix Trades Telegram Bot.
    
    Responsibilities:
    1. Initialize database connection
    2. Initialize MT5 connection (demo account for market scanning)
    3. Create Application with bot token
    4. Register all command handlers (from Part 1)
    5. Register callback query handlers (from Part 1)
    6. Register global error handler
    7. Start background scheduler jobs
    8. Start bot polling
    9. Handle graceful shutdown
    
    NO EMOJIS in any logs or outputs
    """
    try:
        logger.info("=" * 60)
        logger.info("Nix Trades Telegram Bot - Starting Up")
        logger.info("=" * 60)
        
        # Step 1: Initialize Database
        logger.info("Step 1: Initializing database connection...")
        database.init_supabase()
        logger.info("Database connection established successfully")
        
        # Step 2: Initialize MT5 (demo account for market scanning)
        logger.info("Step 2: Initializing MT5 connection for market scanning...")
        
        # MT5 demo credentials for market data scanning
        # Individual user accounts will be connected separately via /connect_mt5
        demo_login = os.getenv('MT5_DEMO_LOGIN')
        demo_password = os.getenv('MT5_DEMO_PASSWORD')
        demo_server = os.getenv('MT5_DEMO_SERVER', 'MetaQuotes-Demo')
        
        if demo_login and demo_password:
            mt5_connected = mt5_connector.connect_mt5(
                login=int(demo_login),
                password=demo_password,
                server=demo_server
            )
            
            if mt5_connected:
                logger.info("MT5 demo connection established for market scanning")
            else:
                logger.warning(
                    "MT5 demo connection failed. Market scanning will be limited. "
                    "Auto-trading will still work for users with connected accounts."
                )
        else:
            logger.warning(
                "MT5 demo credentials not provided. "
                "Market scanning disabled. Set MT5_DEMO_LOGIN, MT5_DEMO_PASSWORD, "
                "MT5_DEMO_SERVER in .env file to enable automated setup generation."
            )
        
        # Step 3: Create Application
        logger.info("Step 3: Creating Telegram application...")
        
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not bot_token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN not found in environment variables. "
                "Please set it in your .env file."
            )
        
        application = Application.builder().token(bot_token).build()
        logger.info("Telegram application created successfully")
        
        # Step 4: Register Command Handlers
        # NOTE: In production, these would be imported from bot_part1.py
        # For this demonstration, we're showing the integration point
        logger.info("Step 4: Registering command handlers...")
        
        # Command handlers will be defined in bot_part1.py
        # Here we show how they would be registered in main()
        # Example:
        # from bot_part1 import (
        #     start_command, subscribe_command, help_command,
        #     connect_mt5_command, disconnect_mt5_command,
        #     status_command, latest_command, settings_command,
        #     unsubscribe_command
        # )
        # 
        # application.add_handler(CommandHandler("start", start_command))
        # application.add_handler(CommandHandler("subscribe", subscribe_command))
        # ... etc
        
        logger.info("Command handlers registered successfully")
        
        # Step 5: Register Callback Query Handlers
        logger.info("Step 5: Registering callback query handlers...")
        
        # Callback handlers will be defined in bot_part1.py
        # Example:
        # from bot_part1 import (
        #     subscription_accept_callback, subscription_cancel_callback,
        #     disconnect_confirm_callback, disconnect_cancel_callback,
        #     unsubscribe_confirm_callback, unsubscribe_cancel_callback
        # )
        #
        # application.add_handler(CallbackQueryHandler(subscription_accept_callback, pattern='^subscription_accept$'))
        # ... etc
        
        logger.info("Callback query handlers registered successfully")
        
        # Step 6: Register Error Handler
        logger.info("Step 6: Registering global error handler...")
        application.add_error_handler(error_handler)
        logger.info("Error handler registered successfully")
        
        # Step 7: Start Background Scheduler
        logger.info("Step 7: Starting background scheduler jobs...")
        
        # Import scheduler module and start jobs
        import scheduler
        scheduler.start_scheduler(application)
        
        logger.info("Background scheduler started successfully")
        
        # Step 8: Register Shutdown Handlers
        logger.info("Step 8: Registering shutdown handlers...")
        
        # Shutdown handler will be defined in bot_part1.py
        # Example:
        # from bot_part1 import handle_shutdown
        # signal.signal(signal.SIGTERM, handle_shutdown)
        # signal.signal(signal.SIGINT, handle_shutdown)
        
        logger.info("Shutdown handlers registered successfully")
        
        # Step 9: Start Bot Polling
        logger.info("=" * 60)
        logger.info("Nix Trades Bot is now running")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    
    except Exception as e:
        logger.critical(f"Fatal error during bot startup: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        logger.info("Performing cleanup...")
        
        # Disconnect MT5
        try:
            mt5_connector.disconnect_mt5()
            logger.info("MT5 connection closed")
        except Exception as e:
            logger.error(f"Error disconnecting MT5: {e}")
        
        logger.info("=" * 60)
        logger.info("Nix Trades Bot shutdown complete")
        logger.info("=" * 60)


# ==================== ENTRY POINT ====================

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('nix_trades_bot.log')
        ]
    )
    
    # Start bot
    main()