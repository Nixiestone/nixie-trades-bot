"""
NIX TRADES - Main Telegram Bot Application
Complete bot with all commands, auto-execution, and graceful shutdown
Production-ready, zero errors, zero placeholders
NO EMOJIS - Professional code only
"""

import logging
import os
import signal
import sys
from typing import Optional
from datetime import datetime

# Load .env file FIRST before any other module reads environment variables
from dotenv import load_dotenv
load_dotenv()

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)

import config
import utils
import database as db
from mt5_connector import MT5Connector
from smc_strategy import SMCStrategy
from ml_models import MLEnsemble
from position_monitor import PositionMonitor
from news_fetcher import NewsFetcher

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler('nix_trades_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class NixTradesBot:
    """
    Main Nix Trades Telegram Bot application.
    Handles all user interactions, trade automation, and system lifecycle.
    """
    
    def __init__(self):
        """Initialize bot components."""
        self.logger = logging.getLogger(f"{__name__}.NixTradesBot")
        
        # Initialize components
        self.application: Optional[Application] = None
        self.mt5 = MT5Connector()
        self.smc = SMCStrategy()
        self.ml = MLEnsemble(mt5_connector=self.mt5)
        self.news = NewsFetcher()
        self.position_monitor: Optional[PositionMonitor] = None
        
        # Graceful shutdown flag
        self.shutting_down = False
        
        self.logger.info("Nix Trades Bot initialized")
    
    # ==================== BOT LIFECYCLE ====================
    
    def run(self):
        """Start the bot."""
        try:
            # Initialize database
            db.init_supabase()
            
            # Create application
            self.application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
            
            # Register command handlers
            self._register_handlers()
            
            # Setup graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.logger.info("Starting Nix Trades Bot...")
            
            # Run bot
            self.application.run_polling(allowed_updates=Update.ALL_TYPES)
        
        except Exception as e:
            self.logger.error(f"Error running bot: {e}")
            sys.exit(1)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        if self.shutting_down:
            return
        
        self.shutting_down = True
        
        self.logger.info(f"Received signal {signum}. Shutting down gracefully...")
        
        # Stop position monitor
        if self.position_monitor:
            self.position_monitor.stop()
        
        # Disconnect MT5
        if self.mt5:
            self.mt5.disconnect()
        
        self.logger.info("Graceful shutdown complete")
        sys.exit(0)
    
    def _register_handlers(self):
        """Register all command and message handlers."""
        app = self.application
        
        # Command handlers
        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("help", self.cmd_help))
        app.add_handler(CommandHandler("subscribe", self.cmd_subscribe))
        app.add_handler(CommandHandler("unsubscribe", self.cmd_unsubscribe))
        app.add_handler(CommandHandler("connect_mt5", self.cmd_connect_mt5))
        app.add_handler(CommandHandler("disconnect_mt5", self.cmd_disconnect_mt5))
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("latest", self.cmd_latest))
        app.add_handler(CommandHandler("settings", self.cmd_settings))
        
        # Callback query handler for inline buttons
        app.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Message handler for text input
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        self.logger.info("Command handlers registered")
    
    # ==================== COMMAND HANDLERS ====================
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        try:
            user = update.effective_user
            
            # Get or create user in database
            db_user = db.get_or_create_user(
                telegram_id=user.id,
                username=user.username,
                first_name=user.first_name,
                timezone='UTC'
            )
            
            if db_user:
                message = utils.replace_forbidden_words(config.START_MESSAGE)
                message = utils.add_footer(message)
                
                await update.message.reply_text(message)
                
                self.logger.info(f"User {user.id} started bot")
            else:
                await update.message.reply_text(
                    "Error creating user account. Please try again later."
                )
        
        except Exception as e:
            self.logger.error(f"Error in /start: {e}")
            await update.message.reply_text("An error occurred. Please try again.")
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        try:
            message = utils.replace_forbidden_words(config.HELP_MESSAGE)
            message = utils.add_footer(message)
            
            await update.message.reply_text(message)
        
        except Exception as e:
            self.logger.error(f"Error in /help: {e}")
            await update.message.reply_text("An error occurred. Please try again.")
    
    async def cmd_subscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /subscribe command."""
        try:
            user_id = update.effective_user.id
            
            # Check if already subscribed
            user = db.get_user(user_id)
            
            if not user:
                await update.message.reply_text("Please use /start first.")
                return
            
            if user.get('subscription_status') == 'active':
                await update.message.reply_text(config.ERROR_MESSAGES['already_subscribed'])
                return
            
            # Show legal disclaimer with confirmation buttons
            keyboard = [
                [
                    InlineKeyboardButton("I Accept", callback_data="subscribe_accept"),
                    InlineKeyboardButton("Cancel", callback_data="subscribe_cancel")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            disclaimer = utils.replace_forbidden_words(config.SUBSCRIPTION_DISCLAIMER)
            
            await update.message.reply_text(disclaimer, reply_markup=reply_markup)
        
        except Exception as e:
            self.logger.error(f"Error in /subscribe: {e}")
            await update.message.reply_text("An error occurred. Please try again.")
    
    async def cmd_unsubscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /unsubscribe command."""
        try:
            user_id = update.effective_user.id
            
            # Update subscription status
            result = db.update_user(user_id, {'subscription_status': 'inactive'})
            
            if result:
                message = utils.add_footer(config.SUCCESS_MESSAGES['unsubscribed'])
                await update.message.reply_text(message)
                
                self.logger.info(f"User {user_id} unsubscribed")
            else:
                await update.message.reply_text("Error updating subscription. Please try again.")
        
        except Exception as e:
            self.logger.error(f"Error in /unsubscribe: {e}")
            await update.message.reply_text("An error occurred. Please try again.")
    
    async def cmd_connect_mt5(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /connect_mt5 command."""
        try:
            user_id = update.effective_user.id
            
            # Send instructions
            instructions = (
                "MT5 ACCOUNT CONNECTION\n\n"
                "Please send your MT5 credentials in this format:\n\n"
                "LOGIN: your_login_number\n"
                "PASSWORD: your_password\n"
                "SERVER: your_broker_server\n\n"
                "Example:\n"
                "LOGIN: 12345678\n"
                "PASSWORD: MyPassword123\n"
                "SERVER: ICMarkets-Demo\n\n"
                "Note: Your password is encrypted and stored securely."
            )
            
            instructions = utils.add_footer(instructions)
            
            # Store state to expect MT5 credentials
            context.user_data['awaiting_mt5_credentials'] = True
            
            await update.message.reply_text(instructions)
        
        except Exception as e:
            self.logger.error(f"Error in /connect_mt5: {e}")
            await update.message.reply_text("An error occurred. Please try again.")
    
    async def cmd_disconnect_mt5(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /disconnect_mt5 command."""
        try:
            user_id = update.effective_user.id
            
            # Remove MT5 credentials from database
            result = db.disconnect_mt5(user_id)
            
            if result:
                message = utils.add_footer(config.SUCCESS_MESSAGES['mt5_disconnected'])
                await update.message.reply_text(message)
                
                self.logger.info(f"User {user_id} disconnected MT5")
            else:
                await update.message.reply_text("Error disconnecting MT5. Please try again.")
        
        except Exception as e:
            self.logger.error(f"Error in /disconnect_mt5: {e}")
            await update.message.reply_text("An error occurred. Please try again.")
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        try:
            user_id = update.effective_user.id
            
            # Get user data
            user = db.get_user(user_id)
            
            if not user:
                await update.message.reply_text("Please use /start first.")
                return
            
            # Build status message
            status_msg = "YOUR STATUS\n\n"
            
            # Subscription
            sub_status = user.get('subscription_status', 'inactive')
            status_msg += f"Subscription: {sub_status.upper()}\n"
            
            # MT5 connection
            mt5_connected = user.get('mt5_login') is not None
            status_msg += f"MT5 Connected: {'YES' if mt5_connected else 'NO'}\n"
            
            # Risk settings
            risk_pct = user.get('risk_percent', config.DEFAULT_RISK_PERCENT)
            status_msg += f"Risk per Trade: {risk_pct}%\n\n"
            
            # Trading statistics
            stats = db.get_user_statistics(user_id)
            
            if stats:
                status_msg += "TRADING STATISTICS\n\n"
                status_msg += f"Total Setups: {stats.get('total_setups', 0)}\n"
                status_msg += f"Successful Trades: {stats.get('successful_trades', 0)}\n"
                status_msg += f"Historical Success Rate: {stats.get('win_rate', 0):.1f}%\n"
                status_msg += f"Total Favorable Outcome: {utils.format_currency(stats.get('total_profit', 0), 'USD')}\n"
                status_msg += f"Average R:R: 1:{stats.get('avg_rr', 0):.2f}\n"
            else:
                status_msg += "No trading history yet.\n"
            
            status_msg = utils.add_footer(status_msg)
            
            await update.message.reply_text(status_msg)
        
        except Exception as e:
            self.logger.error(f"Error in /status: {e}")
            await update.message.reply_text("An error occurred. Please try again.")
    
    async def cmd_latest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /latest command."""
        try:
            user_id = update.effective_user.id
            
            # Check subscription
            user = db.get_user(user_id)
            
            if not user or user.get('subscription_status') != 'active':
                await update.message.reply_text(config.ERROR_MESSAGES['not_subscribed'])
                return
            
            # Get latest setup
            latest_setup = db.get_latest_signal()
            
            if not latest_setup:
                await update.message.reply_text(
                    utils.add_footer("No automated setups available at this time.")
                )
                return
            
            # Format setup message
            setup_msg = self._format_setup_message(latest_setup)
            
            await update.message.reply_text(setup_msg)
        
        except Exception as e:
            self.logger.error(f"Error in /latest: {e}")
            await update.message.reply_text("An error occurred. Please try again.")
    
    async def cmd_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command."""
        try:
            user_id = update.effective_user.id
            
            # Get current settings
            user = db.get_user(user_id)
            
            if not user:
                await update.message.reply_text("Please use /start first.")
                return
            
            risk_pct = user.get('risk_percent', config.DEFAULT_RISK_PERCENT)
            
            # Format settings message
            settings_msg = config.SETTINGS_MESSAGE.format(risk_percent=risk_pct)
            settings_msg = utils.add_footer(settings_msg)
            
            # Store state to expect risk input
            context.user_data['awaiting_risk_input'] = True
            
            await update.message.reply_text(settings_msg)
        
        except Exception as e:
            self.logger.error(f"Error in /settings: {e}")
            await update.message.reply_text("An error occurred. Please try again.")
    
    # ==================== CALLBACK HANDLERS ====================
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks."""
        try:
            query = update.callback_query
            await query.answer()
            
            user_id = query.from_user.id
            data = query.data
            
            if data == "subscribe_accept":
                # Activate subscription
                result = db.update_user(user_id, {'subscription_status': 'active'})
                
                if result:
                    message = utils.add_footer(config.SUCCESS_MESSAGES['subscribed'])
                    await query.edit_message_text(message)
                    
                    self.logger.info(f"User {user_id} subscribed")
                else:
                    await query.edit_message_text("Error activating subscription. Please try again.")
            
            elif data == "subscribe_cancel":
                await query.edit_message_text("Subscription cancelled.")
        
        except Exception as e:
            self.logger.error(f"Error in callback handler: {e}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages (for MT5 credentials and risk input)."""
        try:
            user_id = update.effective_user.id
            text = update.message.text
            
            # Check if awaiting MT5 credentials
            if context.user_data.get('awaiting_mt5_credentials'):
                await self._handle_mt5_credentials(update, context, text)
                context.user_data['awaiting_mt5_credentials'] = False
                return
            
            # Check if awaiting risk input
            if context.user_data.get('awaiting_risk_input'):
                await self._handle_risk_input(update, context, text)
                context.user_data['awaiting_risk_input'] = False
                return
            
            # Unknown message
            await update.message.reply_text(
                "I don't understand that command. Use /help to see available commands."
            )
        
        except Exception as e:
            self.logger.error(f"Error in message handler: {e}")
    
    async def _handle_mt5_credentials(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
        """Parse and save MT5 credentials."""
        try:
            user_id = update.effective_user.id
            
            # Parse credentials
            lines = text.strip().split('\n')
            credentials = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().upper()
                    value = value.strip()
                    
                    if key == 'LOGIN':
                        credentials['login'] = int(value)
                    elif key == 'PASSWORD':
                        credentials['password'] = value
                    elif key == 'SERVER':
                        credentials['server'] = value
            
            if not all(k in credentials for k in ['login', 'password', 'server']):
                await update.message.reply_text(
                    "Invalid format. Please provide LOGIN, PASSWORD, and SERVER."
                )
                return
            
            # Test connection
            await update.message.reply_text("Testing MT5 connection...")
            
            success, message = self.mt5.connect(
                credentials['login'],
                credentials['password'],
                credentials['server']
            )
            
            if success:
                # Save credentials to database (encrypted)
                db.save_mt5_connection(
                    user_id,
                    credentials['login'],
                    credentials['password'],
                    credentials['server']
                )
                
                # Get account info
                account_info = self.mt5.get_account_info()
                
                response = (
                    f"MT5 Connected Successfully\n\n"
                    f"Account: {credentials['login']}\n"
                    f"Server: {credentials['server']}\n"
                    f"Balance: {utils.format_currency(account_info['balance'], account_info['currency'])}\n"
                    f"Currency: {account_info['currency']}\n\n"
                    f"Automated trade execution is now enabled."
                )
                
                response = utils.add_footer(response)
                
                await update.message.reply_text(response)
                
                # Initialize position monitor if not already running
                if not self.position_monitor:
                    self.position_monitor = PositionMonitor(self.mt5, db, self.application.bot)
                    self.position_monitor.start()
                
                self.logger.info(f"User {user_id} connected MT5 account {credentials['login']}")
            
            else:
                await update.message.reply_text(
                    f"MT5 Connection Failed\n\n{message}\n\nPlease check your credentials and try again."
                )
        
        except ValueError:
            await update.message.reply_text("Invalid login number. Please provide a valid numeric login.")
        
        except Exception as e:
            self.logger.error(f"Error handling MT5 credentials: {e}")
            await update.message.reply_text("Error connecting to MT5. Please try again.")
    
    async def _handle_risk_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
        """Parse and save risk percentage."""
        try:
            user_id = update.effective_user.id
            
            # Parse risk percentage
            risk_pct = float(text.strip())
            
            # Validate
            is_valid, error_msg = utils.validate_risk_percent(risk_pct)
            
            if not is_valid:
                await update.message.reply_text(error_msg)
                return
            
            # Update database
            result = db.update_user(user_id, {'risk_percent': risk_pct})
            
            if result:
                message = f"Risk updated to {risk_pct}% per trade."
                message = utils.add_footer(message)
                
                await update.message.reply_text(message)
                
                self.logger.info(f"User {user_id} updated risk to {risk_pct}%")
            else:
                await update.message.reply_text("Error updating risk. Please try again.")
        
        except ValueError:
            await update.message.reply_text(
                "Invalid input. Please send a number between 0.5 and 3.0"
            )
        
        except Exception as e:
            self.logger.error(f"Error handling risk input: {e}")
            await update.message.reply_text("Error updating risk. Please try again.")
    
    # ==================== UTILITY METHODS ====================
    
    def _format_setup_message(self, setup: dict) -> str:
        """
        Format setup alert message.
        
        Args:
            setup: Setup data from database
            
        Returns:
            str: Formatted message
        """
        try:
            symbol = setup['symbol']
            direction = setup['direction']
            entry = setup['entry_price']
            sl = setup['stop_loss']
            tp1 = setup['take_profit_1']
            tp2 = setup['take_profit_2']
            
            # Calculate pips
            sl_pips = utils.calculate_pips(symbol, entry, sl)
            tp1_pips = utils.calculate_pips(symbol, entry, tp1)
            tp2_pips = utils.calculate_pips(symbol, entry, tp2)
            
            # Calculate R:R
            rr_tp1 = utils.calculate_risk_reward(entry, sl, tp1)
            rr_tp2 = utils.calculate_risk_reward(entry, sl, tp2)
            
            # Get current price for order type detection
            current_bid, current_ask = self.mt5.get_current_price(symbol)
            current_price = current_bid if current_bid else entry
            
            order_info = utils.determine_order_type(current_price, entry, symbol)
            
            # Format message
            msg = (
                f"NEW AUTOMATED SETUP: {symbol} | {setup.get('setup_type', 'Standard Setup')}\n"
                f"Model Agreement Score: {setup.get('ml_score', 60)}% {direction}\n\n"
                f"Direction: {direction}\n\n"
                f"Entry: {utils.format_price(symbol, entry)}\n"
                f"Stop Loss: {utils.format_price(symbol, sl)} ({sl_pips:.1f} pips)\n"
                f"TP1: {utils.format_price(symbol, tp1)} (plus {tp1_pips:.1f} pips, R:R {utils.format_risk_reward(rr_tp1)})\n"
                f"TP2: {utils.format_price(symbol, tp2)} (plus {tp2_pips:.1f} pips, R:R {utils.format_risk_reward(rr_tp2)})\n\n"
                f"{order_info['description']}\n\n"
            )
            
            # Add news warning if applicable
            next_news = self.news.get_next_high_impact(symbol)
            if next_news:
                time_until = utils.calculate_time_until(next_news.timestamp)
                msg += f"Note: {next_news.currency} {next_news.title} scheduled in {time_until}\n\n"
            
            msg = utils.replace_forbidden_words(msg)
            msg = utils.add_footer(msg)
            
            return msg
        
        except Exception as e:
            self.logger.error(f"Error formatting setup message: {e}")
            return "Error formatting setup message."


def main():
    """Main entry point."""
    bot = NixTradesBot()
    bot.run()


if __name__ == '__main__':
    main()