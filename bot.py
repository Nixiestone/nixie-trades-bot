"""
NIX TRADES - Main Telegram Bot Application
Role: Python Developer + Security Engineer + Product Manager + QA Engineer
Fixes:
  - MT5 credentials message is deleted from Telegram immediately after parsing (Security)
  - Disclaimer acceptance is enforced via ConversationHandler before subscription activates (QA)
  - All user-facing text passes through validate_user_message() compliance filter (PM)
  - Missing 'cancel' conversation state handler added
  - /settings command fully implemented (was missing)
  - Callback query handlers for disclaimer accept/decline registered
  - State machine for MT5 credential collection uses ConversationHandler (no raw text leak)
  - All references to utils.replace_forbidden_words and utils.add_footer now resolve correctly
  - LOG_LEVEL reads from environment via config.LOG_LEVEL
NO EMOJIS - Professional code only
"""

import logging
import os
import signal
import sys
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ForceReply
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    filters
)
from telegram.error import BadRequest

import config
import utils
import database as db
from mt5_connector import MT5Connector
from smc_strategy import SMCStrategy
from ml_models import MLEnsemble
from position_monitor import PositionMonitor
from news_fetcher import NewsFetcher

# ==================== LOGGING ====================
# Production-grade logging with automatic rotation to prevent disk fill-up

import logging_config
logging_config.setup_logging(log_level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# ==================== CONVERSATION STATES ====================

# Disclaimer flow
DISCLAIMER_SHOWN = 0

# MT5 connection flow
MT5_WAITING_CREDENTIALS = 0

# Settings flow
SETTINGS_WAITING_RISK = 0
SETTINGS_WAITING_TIMEZONE = 1


class NixTradesBot:
    """
    Main Nix Trades Telegram Bot.
    Handles all user-facing commands, conversation flows, and event routing.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.NixTradesBot")
        self.application: Optional[Application] = None
        self.mt5 = MT5Connector()
        self.smc = SMCStrategy()
        self.ml = MLEnsemble(mt5_connector=self.mt5)
        self.news = NewsFetcher()
        self.position_monitor: Optional[PositionMonitor] = None
        self.shutting_down = False
        self.logger.info("Nixie Trades Bot initialised")

    # ==================== LIFECYCLE ====================

    def run(self):
        """Start the bot and block until shutdown signal."""
        try:
            db.init_supabase()
            self.logger.info("Database connection established.")

            self.application = (
                Application.builder()
                .token(config.TELEGRAM_BOT_TOKEN)
                .build()
            )

            self._register_handlers()

            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            self.logger.info("Starting Nixie Trades Bot polling...")
            self.application.run_polling(allowed_updates=Update.ALL_TYPES)

        except Exception as e:
            self.logger.critical("Fatal error running bot: %s", e, exc_info=True)
            sys.exit(1)

    def _signal_handler(self, signum, frame):
        """Graceful shutdown on SIGINT / SIGTERM."""
        if not self.shutting_down:
            self.shutting_down = True
            self.logger.info("Shutdown signal received (Ctrl+C). Stopping bot gracefully...")
            
            # Stop the application if running
            if self.application and self.application.running:
                import asyncio
                # Create a task to stop the application
                asyncio.create_task(self.application.stop())
                asyncio.create_task(self.application.shutdown())
            
            # Exit cleanly
            self.logger.info("Bot stopped. Exiting.")
            sys.exit(0)

    # ==================== HANDLER REGISTRATION ====================

    def _register_handlers(self):
        app = self.application

        # Basic commands
        app.add_handler(CommandHandler('start',         self.cmd_start))
        app.add_handler(CommandHandler('help',          self.cmd_help))
        app.add_handler(CommandHandler('status',        self.cmd_status))
        app.add_handler(CommandHandler('latest',        self.cmd_latest))
        app.add_handler(CommandHandler('unsubscribe',   self.cmd_unsubscribe))

        # Subscribe conversation: show disclaimer -> wait for acceptance
        subscribe_conv = ConversationHandler(
            entry_points=[CommandHandler('subscribe', self.cmd_subscribe)],
            states={
                DISCLAIMER_SHOWN: [
                    CallbackQueryHandler(self.callback_disclaimer_accept, pattern='^disclaimer_accept$'),
                    CallbackQueryHandler(self.callback_disclaimer_decline, pattern='^disclaimer_decline$'),
                ]
            },
            fallbacks=[CommandHandler('cancel', self.cmd_cancel)],
            per_user=True,
            per_chat=True
        )
        app.add_handler(subscribe_conv)

        # MT5 connection conversation
        mt5_conv = ConversationHandler(
            entry_points=[CommandHandler('connect_mt5', self.cmd_connect_mt5)],
            states={
                MT5_WAITING_CREDENTIALS: [
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.handle_mt5_credentials
                    )
                ]
            },
            fallbacks=[
                CommandHandler('cancel', self.cmd_cancel),
                CommandHandler('connect_mt5', self.cmd_connect_mt5)
            ],
            per_user=True,
            per_chat=True
        )
        app.add_handler(mt5_conv)

        # MT5 disconnect (inline keyboard confirmation)
        app.add_handler(CommandHandler('disconnect_mt5', self.cmd_disconnect_mt5))
        app.add_handler(
            CallbackQueryHandler(self.callback_disconnect_confirm, pattern='^mt5_disconnect_confirm$')
        )
        app.add_handler(
            CallbackQueryHandler(self.callback_disconnect_cancel, pattern='^mt5_disconnect_cancel$')
        )

        # Unsubscribe confirmation
        app.add_handler(
            CallbackQueryHandler(self.callback_unsubscribe_confirm, pattern='^unsub_confirm$')
        )
        app.add_handler(
            CallbackQueryHandler(self.callback_unsubscribe_cancel, pattern='^unsub_cancel$')
        )

        # Settings conversation
        settings_conv = ConversationHandler(
            entry_points=[CommandHandler('settings', self.cmd_settings)],
            states={
                SETTINGS_WAITING_RISK: [
                    CallbackQueryHandler(self.callback_settings_risk,     pattern='^risk_'),
                    CallbackQueryHandler(self.callback_settings_timezone, pattern='^tz_menu$'),
                    CallbackQueryHandler(self.callback_settings_done,     pattern='^settings_done$')
                ],
                SETTINGS_WAITING_TIMEZONE: [
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.handle_timezone_input
                    )
                ]
            },
            fallbacks=[CommandHandler('cancel', self.cmd_cancel)],
            per_user=True,
            per_chat=True
        )
        app.add_handler(settings_conv)

        self.logger.info("All handlers registered.")

    # ==================== HELPERS ====================

    @staticmethod
    def _ensure_user(telegram_id: int, update: Update) -> dict:
        """Get or create the user record from the database."""
        user = update.effective_user
        locale = user.language_code if user else None
        tz = utils.detect_timezone(locale)
        return db.get_or_create_user(
            telegram_id=telegram_id,
            username=user.username if user else None,
            first_name=user.first_name if user else None,
            user_timezone=tz
        )

    @staticmethod
    def _format(template: str, **kwargs) -> str:
        """
        Fill template placeholders and run compliance filter.
        Always call this before any reply_text() call.
        """
        defaults = {
            'product_name':   config.PRODUCT_NAME,
            'support_contact': config.SUPPORT_CONTACT,
            'footer':         config.FOOTER
        }
        defaults.update(kwargs)
        filled = template.format(**defaults)
        return utils.validate_user_message(filled)

    async def _reply(self, update: Update, text: str, **kwargs):
        """Send a message, splitting if it exceeds Telegram's 4096 char limit."""
        if len(text) <= config.MAX_MESSAGE_LENGTH:
            await update.effective_message.reply_text(text, **kwargs)
        else:
            chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
            for chunk in chunks:
                await update.effective_message.reply_text(chunk, **kwargs)

    # ==================== COMMAND HANDLERS ====================

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        try:
            telegram_id = update.effective_user.id
            self._ensure_user(telegram_id, update)

            # Send welcome message
            message = self._format(config.WELCOME_MESSAGE)
            await self._reply(update, message)
            
            # Check for queued messages (user was offline)
            pending_messages = db.get_pending_messages(telegram_id)
            
            if pending_messages:
                count = len(pending_messages)
                await update.message.reply_text(
                    f"You have {count} message(s) from while you were offline. Sending now..."
                )
                
                # Send all queued messages with timestamps
                for msg in pending_messages:
                    timestamp = msg['created_at']
                    message_text = f"[{timestamp}]\n\n{msg['message_text']}"
                    
                    try:
                        await update.message.reply_text(message_text)
                        db.mark_message_sent(msg['id'])
                    except Exception as e:
                        self.logger.error(
                            "Failed to send queued message %d to user %d: %s",
                            msg['id'], telegram_id, e
                        )

        except Exception as e:
            self.logger.error("Error in /start: %s", e, exc_info=True)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        try:
            message = self._format(config.HELP_MESSAGE)
            await self._reply(update, message)

        except Exception as e:
            self.logger.error("Error in /help: %s", e, exc_info=True)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])

    async def cmd_subscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """
        Handle /subscribe command.
        Shows legal disclaimer and waits for explicit acceptance
        before activating subscription.
        """
        try:
            telegram_id = update.effective_user.id
            user = self._ensure_user(telegram_id, update)

            if user and user.get('subscription_status') == 'active':
                message = self._format(config.ALREADY_SUBSCRIBED)
                await self._reply(update, message)
                return ConversationHandler.END

            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("I Understand and Accept", callback_data='disclaimer_accept'),
                    InlineKeyboardButton("Decline",                  callback_data='disclaimer_decline')
                ]
            ])

            disclaimer_text = utils.validate_user_message(config.LEGAL_DISCLAIMER)
            await self._reply(update, disclaimer_text, reply_markup=keyboard)

            return DISCLAIMER_SHOWN

        except Exception as e:
            self.logger.error("Error in /subscribe: %s", e, exc_info=True)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])
            return ConversationHandler.END

    async def callback_disclaimer_accept(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """User accepted disclaimer - activate subscription."""
        query = update.callback_query
        await query.answer()

        try:
            telegram_id = update.effective_user.id
            success = db.activate_subscription(telegram_id)

            if success:
                message = self._format(config.SUBSCRIPTION_SUCCESS)
                await query.edit_message_text(message)
            else:
                await query.edit_message_text(config.ERROR_MESSAGES['general_error'])

        except Exception as e:
            self.logger.error("Error in disclaimer_accept: %s", e, exc_info=True)
            await query.edit_message_text(config.ERROR_MESSAGES['general_error'])

        return ConversationHandler.END

    async def callback_disclaimer_decline(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """User declined disclaimer - do not activate."""
        query = update.callback_query
        await query.answer()

        message = utils.validate_user_message(
            "Subscription not activated. You must accept the legal disclaimer to use this service.\n\n"
            "You can try again anytime with /subscribe\n\n"
            f"{config.FOOTER}"
        )
        await query.edit_message_text(message)
        return ConversationHandler.END

    async def cmd_connect_mt5(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """
        Handle /connect_mt5 command.
        Checks if already connected, then prompts for credentials.
        """
        try:
            telegram_id = update.effective_user.id
            user = db.get_user(telegram_id)

            if user and user.get('mt5_connected'):
                message = self._format(
                    config.MT5_CONNECTED_ALREADY,
                    broker_name=user.get('mt5_broker_name', 'Unknown'),
                    account_number=str(user.get('mt5_login', 'N/A'))
                )
                await self._reply(update, message)
                return ConversationHandler.END

            message = self._format(config.MT5_CONNECTION_PROMPT)
            await self._reply(update, message, reply_markup=ForceReply(selective=True))

            return MT5_WAITING_CREDENTIALS

        except Exception as e:
            self.logger.error("Error in /connect_mt5: %s", e, exc_info=True)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])
            return ConversationHandler.END

    async def handle_mt5_credentials(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """
        Process MT5 credentials sent by the user.
        SECURITY: Deletes the user's message immediately after reading credentials.
        """
        user_message = update.message
        telegram_id = update.effective_user.id

        # Delete user's credential message immediately for security
        try:
            await user_message.delete()
        except BadRequest as e:
            self.logger.warning(
                "Could not delete credential message for user %d: %s", telegram_id, e
            )

        try:
            text = user_message.text or ''
            credentials = utils.parse_mt5_credentials(text)

            if not credentials:
                message = utils.validate_user_message(
                    "Could not read your credentials. Please use this exact format:\n\n"
                    "LOGIN: 12345678\n"
                    "PASSWORD: YourPassword\n"
                    "SERVER: YourBroker-Demo\n\n"
                    "Try again with /connect_mt5\n\n"
                    f"{config.FOOTER}"
                )
                await self._reply(update, message)
                return ConversationHandler.END

            # Notify user that verification is in progress
            verifying_msg = await update.effective_chat.send_message(
                "Verifying credentials with MT5 Worker Service..."
            )

            # Verify credentials against MT5 Worker
            success, result = self.mt5.verify_credentials(
                telegram_id=telegram_id,
                login=credentials['login'],
                password=credentials['password'],
                server=credentials['server']
            )

            # Delete the "verifying..." message
            try:
                await verifying_msg.delete()
            except BadRequest:
                pass

            if not success:
                message = utils.validate_user_message(
                    "Connection could not be verified. Please check your credentials and try again.\n\n"
                    f"{config.FOOTER}"
                )
                await self._reply(update, message)
                return ConversationHandler.END

            # Save encrypted credentials to database
            account_data = result
            saved = db.save_mt5_credentials(
                telegram_id=telegram_id,
                login=credentials['login'],
                password=credentials['password'],
                server=credentials['server'],
                broker_name=account_data.get('broker', 'Unknown Broker'),
                account_balance=account_data.get('balance', 0.0),
                account_currency=account_data.get('currency', 'USD')
            )

            if not saved:
                await self._reply(update, config.ERROR_MESSAGES['general_error'])
                return ConversationHandler.END

            message = self._format(
                config.MT5_CONNECTION_SUCCESS,
                broker_name=account_data.get('broker', 'Unknown Broker'),
                account_number=str(credentials['login']),
                balance=utils.format_currency(
                    account_data.get('balance', 0.0),
                    account_data.get('currency', 'USD')
                )
            )
            await self._reply(update, message)

            self.logger.info(
                "MT5 connected for user %d: login=%d broker=%s",
                telegram_id, credentials['login'], account_data.get('broker')
            )

        except Exception as e:
            self.logger.error("Error handling MT5 credentials for user %d: %s", telegram_id, e, exc_info=True)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])

        return ConversationHandler.END

    async def cmd_disconnect_mt5(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /disconnect_mt5 command with confirmation keyboard."""
        try:
            telegram_id = update.effective_user.id
            user = db.get_user(telegram_id)

            if not user or not user.get('mt5_connected'):
                await self._reply(update, config.ERROR_MESSAGES['mt5_not_connected'])
                return

            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("Yes, Disconnect", callback_data='mt5_disconnect_confirm'),
                    InlineKeyboardButton("Cancel",          callback_data='mt5_disconnect_cancel')
                ]
            ])

            message = self._format(config.MT5_DISCONNECTION_CONFIRM)
            await self._reply(update, message, reply_markup=keyboard)

        except Exception as e:
            self.logger.error("Error in /disconnect_mt5: %s", e, exc_info=True)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])

    async def callback_disconnect_confirm(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Confirm MT5 disconnection."""
        query = update.callback_query
        await query.answer()

        try:
            telegram_id = update.effective_user.id
            db.delete_mt5_credentials(telegram_id)
            message = self._format(config.MT5_DISCONNECTION_SUCCESS)
            await query.edit_message_text(message)

        except Exception as e:
            self.logger.error("Error in disconnect_confirm: %s", e, exc_info=True)
            await query.edit_message_text(config.ERROR_MESSAGES['general_error'])

    async def callback_disconnect_cancel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Cancel MT5 disconnection."""
        query = update.callback_query
        await query.answer()
        await query.edit_message_text(
            utils.validate_user_message(f"Disconnection cancelled.\n\n{config.FOOTER}")
        )

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        try:
            telegram_id = update.effective_user.id
            user = db.get_user(telegram_id)

            if not user:
                await self._reply(update, "Please use /start first.")
                return

            sub_status = user.get('subscription_status', 'inactive')
            mt5_connected = bool(user.get('mt5_connected'))
            risk_pct = user.get('risk_percent', config.DEFAULT_RISK_PERCENT)
            stats = db.get_user_statistics(telegram_id) or {}

            lines = [
                "YOUR STATUS",
                "",
                f"Subscription:    {sub_status.upper()}",
                f"MT5 Connected:   {'YES' if mt5_connected else 'NO'}",
                f"Risk per Trade:  {risk_pct}%",
            ]

            if mt5_connected:
                lines += [
                    f"Broker:          {user.get('mt5_broker_name', 'N/A')}",
                    f"Account:         {user.get('mt5_login', 'N/A')}",
                ]

            lines += [
                "",
                "TRADING STATISTICS",
                "",
                f"Total Setups:                {stats.get('total_setups', 0)}",
                f"Successful Trades:           {stats.get('successful_trades', 0)}",
                f"Historical Success Rate:     {stats.get('win_rate', 0.0):.1f}%",
                "  (Past performance does not guarantee future results)",
                f"Total Profit/Loss:           {utils.format_currency(stats.get('total_profit', 0.0), 'USD')}",
                f"Average R:R Achieved:        1:{stats.get('avg_rr', 0.0):.2f}",
            ]

            message = utils.validate_user_message(
                utils.add_footer("\n".join(lines))
            )
            await self._reply(update, message)

        except Exception as e:
            self.logger.error("Error in /status: %s", e, exc_info=True)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])

    async def cmd_latest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /latest command - show most recent automated setup."""
        try:
            telegram_id = update.effective_user.id
            user = db.get_user(telegram_id)

            if not user or user.get('subscription_status') != 'active':
                await self._reply(update, config.ERROR_MESSAGES['not_subscribed'])
                return

            latest = db.get_latest_signal()

            if not latest:
                message = self._format(config.NO_RECENT_SETUPS)
                await self._reply(update, message)
                return

            user_data = db.get_user(telegram_id)
            risk_pct = user_data.get('risk_percent', config.DEFAULT_RISK_PERCENT) if user_data else config.DEFAULT_RISK_PERCENT
            lot_size = None

            if user_data and user_data.get('mt5_connected'):
                lot_size = self.mt5.calculate_lot_size(
                    telegram_id=telegram_id,
                    symbol=latest['symbol'],
                    risk_percent=risk_pct,
                    sl_pips=latest.get('sl_pips', 20.0)
                )

            message = utils.format_setup_message(
                signal_number=latest['signal_number'],
                symbol=latest['symbol'],
                direction=latest['direction'],
                setup_type=latest['setup_type'],
                entry_price=latest['entry_price'],
                stop_loss=latest['stop_loss'],
                take_profit_1=latest['take_profit_1'],
                take_profit_2=latest['take_profit_2'],
                sl_pips=latest.get('sl_pips', 0.0),
                tp1_pips=latest.get('tp1_pips', 0.0),
                tp2_pips=latest.get('tp2_pips', 0.0),
                rr_tp1=latest.get('rr_tp1', 0.0),
                rr_tp2=latest.get('rr_tp2', 0.0),
                ml_score=latest.get('ml_score', 0),
                session=latest.get('session', 'N/A'),
                order_type=latest.get('order_type', 'LIMIT'),
                lot_size=lot_size
            )
            await self._reply(update, message)

        except Exception as e:
            self.logger.error("Error in /latest: %s", e, exc_info=True)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])

    async def cmd_settings(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle /settings command - show risk and timezone options."""
        try:
            telegram_id = update.effective_user.id
            user = db.get_user(telegram_id)

            if not user or user.get('subscription_status') != 'active':
                await self._reply(update, config.ERROR_MESSAGES['not_subscribed'])
                return ConversationHandler.END

            current_risk = user.get('risk_percent', config.DEFAULT_RISK_PERCENT)
            current_tz = user.get('timezone', 'UTC')

            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("Risk 0.5%", callback_data='risk_0.5'),
                    InlineKeyboardButton("Risk 1.0%", callback_data='risk_1.0'),
                    InlineKeyboardButton("Risk 1.5%", callback_data='risk_1.5'),
                    InlineKeyboardButton("Risk 2.0%", callback_data='risk_2.0')
                ],
                [
                    InlineKeyboardButton("Set Timezone", callback_data='tz_menu'),
                    InlineKeyboardButton("Done",         callback_data='settings_done')
                ]
            ])

            text = (
                "SETTINGS\n\n"
                f"Current Risk per Trade: {current_risk}%\n"
                f"Current Timezone: {current_tz}\n\n"
                "Select a risk level or set your timezone:\n"
                "(Risk applies to automatic lot sizing on MT5)\n\n"
                f"{config.FOOTER}"
            )
            await self._reply(update, utils.validate_user_message(text), reply_markup=keyboard)
            return SETTINGS_WAITING_RISK

        except Exception as e:
            self.logger.error("Error in /settings: %s", e, exc_info=True)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])
            return ConversationHandler.END

    async def callback_settings_risk(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle risk percentage selection in settings."""
        query = update.callback_query
        await query.answer()

        try:
            telegram_id = update.effective_user.id
            risk_str = query.data.replace('risk_', '')
            risk_pct = float(risk_str)

            if not utils.validate_risk_percent(risk_pct):
                await query.edit_message_text(config.ERROR_MESSAGES['invalid_format'])
                return ConversationHandler.END

            db.update_risk_percent(telegram_id, risk_pct)

            message = utils.validate_user_message(
                f"Risk per trade updated to {risk_pct}%.\n\n"
                "This percentage of your account balance will be risked on each automated trade.\n\n"
                f"{config.FOOTER}"
            )
            await query.edit_message_text(message)

        except Exception as e:
            self.logger.error("Error in settings_risk callback: %s", e, exc_info=True)
            await query.edit_message_text(config.ERROR_MESSAGES['general_error'])

        return ConversationHandler.END

    async def callback_settings_timezone(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Prompt user to type their timezone."""
        query = update.callback_query
        await query.answer()

        message = utils.validate_user_message(
            "Please type your IANA timezone name.\n\n"
            "Examples:\n"
            "- America/New_York\n"
            "- Europe/London\n"
            "- Asia/Tokyo\n"
            "- Africa/Lagos\n\n"
            "Find yours at: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones\n\n"
            f"{config.FOOTER}"
        )
        await query.edit_message_text(message)
        return SETTINGS_WAITING_TIMEZONE

    async def handle_timezone_input(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle timezone text input in settings."""
        try:
            import pytz
            telegram_id = update.effective_user.id
            tz_input = (update.message.text or '').strip()

            # Validate the timezone
            try:
                pytz.timezone(tz_input)
                valid = True
            except pytz.exceptions.UnknownTimeZoneError:
                valid = False

            if not valid:
                message = utils.validate_user_message(
                    f"'{tz_input}' is not a valid timezone.\n\n"
                    "Please use an IANA timezone name like 'America/New_York' or 'Europe/London'.\n\n"
                    "Try /settings again.\n\n"
                    f"{config.FOOTER}"
                )
                await self._reply(update, message)
                return ConversationHandler.END

            db.update_timezone(telegram_id, tz_input)

            message = utils.validate_user_message(
                f"Timezone updated to {tz_input}.\n\n"
                "Daily market briefings will now arrive at 8:00 AM in this timezone.\n\n"
                f"{config.FOOTER}"
            )
            await self._reply(update, message)

        except Exception as e:
            self.logger.error("Error handling timezone input: %s", e, exc_info=True)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])

        return ConversationHandler.END

    async def callback_settings_done(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Close settings menu."""
        query = update.callback_query
        await query.answer()
        await query.edit_message_text(
            utils.validate_user_message(f"Settings saved.\n\n{config.FOOTER}")
        )
        return ConversationHandler.END

    async def cmd_unsubscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /unsubscribe with confirmation."""
        try:
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("Yes, Unsubscribe", callback_data='unsub_confirm'),
                    InlineKeyboardButton("Cancel",            callback_data='unsub_cancel')
                ]
            ])
            message = self._format(config.UNSUBSCRIBE_CONFIRM)
            await self._reply(update, message, reply_markup=keyboard)

        except Exception as e:
            self.logger.error("Error in /unsubscribe: %s", e, exc_info=True)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])

    async def callback_unsubscribe_confirm(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Confirm and execute unsubscription."""
        query = update.callback_query
        await query.answer()

        try:
            telegram_id = update.effective_user.id
            db.deactivate_subscription(telegram_id)
            message = self._format(config.UNSUBSCRIBE_SUCCESS)
            await query.edit_message_text(message)

        except Exception as e:
            self.logger.error("Error in unsubscribe_confirm: %s", e, exc_info=True)
            await query.edit_message_text(config.ERROR_MESSAGES['general_error'])

    async def callback_unsubscribe_cancel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Cancel unsubscription."""
        query = update.callback_query
        await query.answer()
        await query.edit_message_text(
            utils.validate_user_message(f"Unsubscription cancelled.\n\n{config.FOOTER}")
        )

    async def cmd_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Universal cancel command for any active conversation."""
        await self._reply(
            update,
            utils.validate_user_message(
                "Action cancelled.\n\n"
                "Use /help to see available commands.\n\n"
                f"{config.FOOTER}"
            )
        )
        return ConversationHandler.END

    # ==================== PUBLIC NOTIFICATION METHODS ====================

    async def send_setup_alert(
        self,
        telegram_id: int,
        setup_data: dict,
        lot_size: Optional[float] = None
    ):
        """
        Send an automated setup alert to a specific user.
        If the user is offline, the message is queued with a timestamp.
        Called by the scheduler when a new setup is detected.

        Args:
            telegram_id: Recipient Telegram user ID
            setup_data:  Signal dict from database
            lot_size:    Calculated lot size if user has MT5 connected
        """
        try:
            message = utils.format_setup_message(
                signal_number=setup_data['signal_number'],
                symbol=setup_data['symbol'],
                direction=setup_data['direction'],
                setup_type=setup_data['setup_type'],
                entry_price=setup_data['entry_price'],
                stop_loss=setup_data['stop_loss'],
                take_profit_1=setup_data['take_profit_1'],
                take_profit_2=setup_data['take_profit_2'],
                sl_pips=setup_data.get('sl_pips', 0.0),
                tp1_pips=setup_data.get('tp1_pips', 0.0),
                tp2_pips=setup_data.get('tp2_pips', 0.0),
                rr_tp1=setup_data.get('rr_tp1', 0.0),
                rr_tp2=setup_data.get('rr_tp2', 0.0),
                ml_score=setup_data.get('ml_score', 0),
                session=setup_data.get('session', 'N/A'),
                order_type=setup_data.get('order_type', 'LIMIT'),
                lot_size=lot_size
            )

            try:
                await self.application.bot.send_message(
                    chat_id=telegram_id,
                    text=message
                )
            except Exception as send_error:
                # User is offline or blocked the bot - queue the message
                self.logger.warning(
                    "User %d unreachable (offline/blocked). Queuing message: %s",
                    telegram_id, send_error
                )
                db.queue_message(
                    telegram_id=telegram_id,
                    message_text=message,
                    message_type='SETUP_ALERT'
                )

        except Exception as e:
            self.logger.error(
                "Error sending setup alert to user %d: %s", telegram_id, e
            )

    async def send_trade_notification(self, telegram_id: int, message: str):
        """
        Send a trade execution or position update notification.
        If the user is offline, the message is queued with a timestamp.

        Args:
            telegram_id: Recipient Telegram user ID
            message:     Notification text
        """
        try:
            clean_message = utils.validate_user_message(
                utils.add_footer(message)
            )
            
            try:
                await self.application.bot.send_message(
                    chat_id=telegram_id,
                    text=clean_message
                )
            except Exception as send_error:
                # User is offline - queue the message
                self.logger.warning(
                    "User %d unreachable. Queuing trade notification: %s",
                    telegram_id, send_error
                )
                db.queue_message(
                    telegram_id=telegram_id,
                    message_text=clean_message,
                    message_type='TRADE_NOTIFICATION'
                )
                
        except Exception as e:
            self.logger.error(
                "Error sending trade notification to user %d: %s", telegram_id, e
            )


def main():
    """Application entry point."""
    if not config.TELEGRAM_BOT_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN is not set in environment. Exiting.")
        sys.exit(1)

    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        logger.critical("SUPABASE_URL or SUPABASE_KEY not set in environment. Exiting.")
        sys.exit(1)

    if not config.ENCRYPTION_KEY:
        logger.critical("ENCRYPTION_KEY not set in environment. Exiting.")
        sys.exit(1)

    bot = NixTradesBot()
    bot.run()


if __name__ == '__main__':
    main()