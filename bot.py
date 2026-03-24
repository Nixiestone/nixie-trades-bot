import asyncio
import io
import logging
import sys
from typing import Optional
from datetime import datetime, timezone
from collections import defaultdict
import time as time_module

from dotenv import load_dotenv
load_dotenv()

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)
from telegram.error import BadRequest, Forbidden, NetworkError, TelegramError
from telegram.request import HTTPXRequest

import config
import utils
import database as db
from mt5_connector import MT5Connector
from chart_generator import ChartGenerator as _ChartGenerator

# Module-level cache for the sample chart PNG.
# Generated once on first /start or /help call and reused thereafter.
_SAMPLE_CHART_BYTES: Optional[bytes] = None
from payment_handler import get_subscription_manager
from smc_strategy import SMCStrategy
from ml_models import MLEnsemble
from position_monitor import PositionMonitor
from news_fetcher import NewsFetcher
from scheduler import NixTradesScheduler

import logging_config
logging_config.setup_logging(log_level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# ==================== CONVERSATION STATES ====================

DISCLAIMER_SHOWN           = 0
MT5_WAITING_LOGIN          = 0
MT5_WAITING_PASSWORD       = 1
MT5_WAITING_SERVER         = 2
SETTINGS_WAITING_RISK      = 0
SETTINGS_WAITING_TZ        = 1
AUTO_MGMT_DISCLAIMER_SHOWN = 0

# ==================== RATE LIMITER ====================

class _RateLimiter:
    """
    Simple in-memory rate limiter: 20 commands per 60 seconds per user.
    Prevents abuse and protects the bot from being spammed.
    """
    MAX_CALLS  = 20
    PERIOD_SEC = 60

    def __init__(self):
        self._calls: dict = defaultdict(list)

    def is_allowed(self, user_id: int) -> bool:
        now   = time_module.monotonic()
        calls = self._calls[user_id]
        # Remove calls older than the window
        self._calls[user_id] = [t for t in calls if now - t < self.PERIOD_SEC]
        if len(self._calls[user_id]) >= self.MAX_CALLS:
            return False
        self._calls[user_id].append(now)
        return True

_rate_limiter = _RateLimiter()

# ==================== SEND RETRY SETTINGS ====================

_SEND_MAX_ATTEMPTS = 4
_SEND_BASE_DELAY   = 1.0  # seconds; doubles each attempt

# ==================== SAMPLE NOTIFICATIONS ====================

SAMPLE_SETUP_ALERT = """AUTOMATED SETUP ALERT - SAMPLE

Setup Number:  001
Symbol:        EURUSD
Direction:     BUY
Setup Type:    Unicorn Breaker Block
Ensemble Score: 78%
Session:       London

Entry Price:   1.08450  (Limit Order - price must pull back to this level)
Stop Loss:     1.08150  (-30.0 pips)
Take Profit 1: 1.08900  (+45.0 pips, R:R 1:1.5)
Take Profit 2: 1.09200  (+75.0 pips, R:R 1:2.5)

Order Type: LIMIT ORDER
Price needs to pull back to the entry zone before the order fills.
The order will expire in 8 hours if not filled.

Position Management (Automatic if MT5 connected):
  - At TP1: 50% of position closed, stop loss moved to breakeven
  - At TP2: Remaining 50% closed, trade complete

News Alert: USD CPI scheduled in 2 hours 15 minutes.
Caution is advised around this event.

EDUCATIONAL PURPOSES ONLY. NOT FINANCIAL ADVICE.
Past performance does not guarantee future results.
Nixie Trades | Smart Money, Automated Logic"""

SAMPLE_NEWS_ALERT = """DAILY MARKET OVERVIEW - SAMPLE
Tuesday, February 25, 2026 (UTC)

MARKET STRUCTURE:
  EURUSD: BULLISH (72% confidence) - Higher highs and higher lows confirmed on Daily
  GBPUSD: BULLISH (65% confidence) - Broken above key resistance at 1.2650
  USDJPY: BEARISH (68% confidence) - Rejection from 152.00 resistance zone
  XAUUSD: BULLISH (74% confidence) - Strong demand above 2880.00

HIGH-IMPACT NEWS TODAY:
  08:30 UTC  USD   Initial Jobless Claims     HIGH IMPACT
  13:30 UTC  USD   Core CPI Month-over-Month  HIGH IMPACT
  15:00 UTC  USD   Fed Chair Speech           HIGH IMPACT

Trading paused automatically 30 minutes before and 15 minutes after each event.

The system scans all pairs every 15 minutes. Qualifying setups will be sent when detected.

EDUCATIONAL PURPOSES ONLY. NOT FINANCIAL ADVICE.
Nixie Trades | Smart Money, Automated Logic"""


AUTO_MGMT_DISCLAIMER = (
    "AUTONOMOUS POSITION MANAGEMENT - RISK DISCLOSURE\n\n"
    "You are about to enable autonomous position management.\n\n"
    "What this means:\n"
    "When enabled, Nixie Trades will automatically:\n"
    "  - Close 50 percent of your position when TP1 is reached\n"
    "  - Move your stop loss to breakeven after TP1\n"
    "  - Close the remaining 50 percent when TP2 is reached\n\n"
    "No confirmation message will be sent to you before each action.\n"
    "The bot will act immediately and notify you after the fact.\n\n"
    "What this does NOT change:\n"
    "  - You retain full control of your MetaTrader 5 account at all times\n"
    "  - You can disconnect the bot or close trades manually at any time\n"
    "  - You can disable this setting again at any time via /settings\n\n"
    "ACKNOWLEDGMENT\n"
    "By tapping 'Enable Autonomous Management', you confirm:\n"
    "  - You authorise Nixie Trades to modify and partially close positions "
    "on your account without asking for confirmation each time\n"
    "  - You understand this is an automated educational tool, not financial advice\n"
    "  - You accept full responsibility for all outcomes\n\n"
    f"{config.FOOTER}"
)


class NixTradesBot:
    """
    Main Nix Trades Telegram Bot.
    """

    def __init__(self):
        self.logger            = logging.getLogger(f"{__name__}.NixTradesBot")
        self.application: Optional[Application] = None
        self.mt5               = MT5Connector()
        self.smc               = SMCStrategy()
        self.ml                = MLEnsemble(mt5_connector=self.mt5)
        self.news              = NewsFetcher()
        self.position_monitor: Optional[PositionMonitor] = None
        self.scheduler_obj: Optional[NixTradesScheduler] = None
        self.logger.info("Nixie Trades Bot initialised")

    # ==================== LIFECYCLE ====================

    def run(self):
        """
        Entry point. Delegates all async work to asyncio.run() which creates
        one persistent event loop for the entire bot lifetime. The retry loop
        runs inside that single loop, which eliminates the 'Event loop is closed'
        error that occurs when run_polling() is called more than once.
        """
        try:
            db.init_supabase()
            self.logger.info("Database connection established.")
        except Exception as e:
            self.logger.critical("Database initialisation failed: %s", e, exc_info=True)
            sys.exit(1)

        if not config.TELEGRAM_BOT_TOKEN:
            self.logger.critical("TELEGRAM_BOT_TOKEN is not set in your .env file.")
            sys.exit(1)

        try:
            asyncio.run(self._main())
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by keyboard interrupt.")

    async def _main(self):
        """
        Persistent async retry loop running inside the single event loop
        created by asyncio.run(). A failed connection attempt never closes
        the loop so every retry receives a clean, live event loop.

        Retry delays (seconds): 15, 30, 60, 120, 300, then capped at 300.
        The attempt counter resets to 0 after every successful connection
        so reconnections after a network drop always start from 15 seconds.
        """
        _attempt    = 0
        _base_delay = 15
        _max_delay  = 300

        while True:
            _attempt += 1
            _delay    = min(_base_delay * _attempt, _max_delay)
            self.application    = None
            _app_started        = False
            _polling_started    = False

            try:
                _request = HTTPXRequest(
                    connection_pool_size=8,
                    connect_timeout=20.0,
                    read_timeout=30.0,
                    write_timeout=30.0,
                    pool_timeout=15.0,
                )
                # Build without PTB post_init/post_stop hooks.
                # Lifecycle is managed manually below for full retry control.
                self.application = (
                    Application.builder()
                    .token(config.TELEGRAM_BOT_TOKEN)
                    .request(_request)
                    .build()
                )
                self._register_handlers()
                self.logger.info(
                    "Connecting to Telegram (attempt %d, "
                    "next retry delay if this fails: %ds)...",
                    _attempt, _delay,
                )

                # async with calls application.initialize() on entry which
                # performs the getMe() network call that validates the token.
                # If the network is down this raises and we go to the except block.
                async with self.application:
                    try:
                        await self._post_init(self.application)
                        await self.application.start()
                        _app_started = True

                        await self.application.updater.start_polling(
                            allowed_updates=Update.ALL_TYPES,
                            drop_pending_updates=(_attempt == 1),
                        )
                        _polling_started = True

                        self.logger.info(
                            "Nixie Trades Bot is running. "
                            "Connection established on attempt %d.",
                            _attempt,
                        )
                        # Reset counter so next reconnect starts from 15s
                        _attempt = 0

                        # Block here until Ctrl+C cancels this coroutine
                        await asyncio.Event().wait()

                    except (asyncio.CancelledError, KeyboardInterrupt):
                        self.logger.info(
                            "Shutdown signal received. Stopping cleanly...")

                    finally:
                        if _polling_started:
                            try:
                                await self.application.updater.stop()
                            except Exception as _ue:
                                self.logger.debug(
                                    "Updater stop error (non-critical): %s", _ue)
                        if _app_started:
                            try:
                                await self._post_stop(self.application)
                                await self.application.stop()
                            except Exception as _ae:
                                self.logger.debug(
                                    "Application stop error (non-critical): %s", _ae)

                # Reached only after a clean Ctrl+C shutdown
                if _polling_started:
                    return

            except (asyncio.CancelledError, KeyboardInterrupt):
                return

            except (TelegramError, NetworkError, OSError) as e:
                self.logger.warning(
                    "Connection failed (attempt %d): %s. "
                    "Retrying automatically in %d seconds. "
                    "The bot will keep retrying until the network is available.",
                    _attempt, e, _delay,
                )
                await asyncio.sleep(_delay)

            except Exception as e:
                self.logger.error(
                    "Unexpected error on attempt %d: %s. "
                    "Retrying in %d seconds.",
                    _attempt, e, _delay,
                    exc_info=True,
                )
                await asyncio.sleep(_delay)

    async def _post_init(self, application: Application):
        """
        Called after the Application is fully initialised and the event loop
        is running. Safe to start background services here.
        """
        # Start position monitor (runs in its own background thread)
        self.position_monitor = PositionMonitor(
            mt5_connector=self.mt5,
            database=db,
            telegram_bot=application.bot,
        )
        self.position_monitor.start()

        # Make position_monitor accessible to all callback handlers via
        # context.bot_data. This is the correct PTB pattern for sharing
        # application-level state with handler functions.
        application.bot_data['position_monitor'] = self.position_monitor

        # Start the scheduler (APScheduler, shares the running event loop)
        self.scheduler_obj = NixTradesScheduler(
            telegram_bot=application.bot,
            mt5_connector=self.mt5,
            smc_strategy=self.smc,
            ml_ensemble=self.ml,
            news_fetcher=self.news,
            position_monitor=self.position_monitor,
        )
        self.scheduler_obj.start()

        # Pre-connect all registered MT5 accounts so first trades are instant
        await self.mt5.connect_all_users()
        self.logger.info("Background services started: position monitor and market scanner.")

    async def _post_stop(self, application: Application):
        """
        Stops all background services.
        Called manually from _main() before application.stop().
        Sets references to None after stopping so repeated calls are safe.
        """
        if self.scheduler_obj is None and self.position_monitor is None:
            return
        self.logger.info("Shutting down background services...")
        if self.scheduler_obj:
            self.scheduler_obj.stop()
            self.scheduler_obj = None
        if self.position_monitor:
            self.position_monitor.stop()
            self.position_monitor = None
        self.logger.info("All services stopped.")

    # ==================== HANDLER REGISTRATION ====================

    def _register_handlers(self):
        app = self.application

        # ---- Global error handler (catches network errors, etc.) ----
        app.add_error_handler(self._error_handler)

        # ---- Simple commands ----
        app.add_handler(CommandHandler('start',          self.cmd_start))
        app.add_handler(CommandHandler('help',           self.cmd_help))
        app.add_handler(CommandHandler('status',         self.cmd_status))
        app.add_handler(CommandHandler('latest',         self.cmd_latest))
        app.add_handler(CommandHandler('unsubscribe',    self.cmd_unsubscribe))
        app.add_handler(CommandHandler('disconnect_mt5', self.cmd_disconnect_mt5))
        app.add_handler(CommandHandler('download',      self.cmd_download))

        # ---- Subscribe conversation ----
        # per_message=True is required when CallbackQueryHandler is inside a
        # ConversationHandler to suppress the PTBUserWarning.
        subscribe_conv = ConversationHandler(
            entry_points=[CommandHandler('subscribe', self.cmd_subscribe)],
            states={
                DISCLAIMER_SHOWN: [
                    CallbackQueryHandler(
                        self.callback_disclaimer_accept, pattern='^disclaimer_accept$'
                    ),
                    CallbackQueryHandler(
                        self.callback_disclaimer_decline, pattern='^disclaimer_decline$'
                    ),
                ],
            },
            fallbacks=[CommandHandler('cancel', self.cmd_cancel)],
            per_user=True,
            per_chat=True,
            per_message=False,
        )
        app.add_handler(subscribe_conv)

        # ---- MT5 connection conversation (step by step, no raw credential message) ----
        mt5_conv = ConversationHandler(
            entry_points=[CommandHandler('connect_mt5', self.cmd_connect_mt5)],
            states={
                MT5_WAITING_LOGIN: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_mt5_login),
                ],
                MT5_WAITING_PASSWORD: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_mt5_password),
                ],
                MT5_WAITING_SERVER: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_mt5_server),
                ],
            },
            fallbacks=[CommandHandler('cancel', self.cmd_cancel)],
            per_user=True,
            per_chat=True,
        )
        app.add_handler(mt5_conv)

        # ---- Settings conversation ----
        # entry_points has TWO triggers:
        #   1. /settings command -> shows keyboard
        #   2. "Set Timezone" button press -> opens TZ text input state
        settings_conv = ConversationHandler(
            entry_points=[
                CommandHandler('settings', self.cmd_settings),
                CallbackQueryHandler(
                    self.callback_settings_tz_prompt, pattern='^settings_tz_prompt$'),
            ],
            states={},
            fallbacks=[CommandHandler('cancel', self.cmd_cancel)],
            per_user=True,
            per_chat=True,
            per_message=False,
        )
        app.add_handler(settings_conv)

        # ---- Standalone callback buttons ----
        app.add_handler(CallbackQueryHandler(
            self.callback_unsubscribe_confirm, pattern='^unsub_confirm$'
        ))
        app.add_handler(CallbackQueryHandler(
            self.callback_unsubscribe_cancel, pattern='^unsub_cancel$'
        ))
        app.add_handler(CallbackQueryHandler(
            self.callback_disconnect_confirm, pattern='^mt5_disconnect_confirm$'
        ))
        app.add_handler(CallbackQueryHandler(
            self.callback_disconnect_cancel, pattern='^mt5_disconnect_cancel$'
        ))
        app.add_handler(CallbackQueryHandler(
            self.callback_mt5_reconnect_confirm, pattern='^mt5_reconnect_confirm$'
        ))
        app.add_handler(CallbackQueryHandler(
            self.callback_mt5_reconnect_cancel, pattern='^mt5_reconnect_cancel$'
        ))
        app.add_handler(CallbackQueryHandler(
            self.callback_settings_risk, pattern=r'^risk_\d+\.?\d*$'
        ))
        app.add_handler(CallbackQueryHandler(
            self.callback_settings_done, pattern='^settings_done$'
        ))
        app.add_handler(CallbackQueryHandler(
            self.callback_settings_tz_select, pattern=r'^tz_sel_'
        ))
        app.add_handler(CallbackQueryHandler(
            self.callback_partial_close,
            pattern=r'^partial_close_(yes|no)_\d+$'
        ))
        app.add_handler(CallbackQueryHandler(
            self.callback_breakeven_decision,
            pattern=r'^breakeven_(yes|no)_\d+$'
        ))
        # ---- Auto position management disclaimer conversation ----
        auto_mgmt_conv = ConversationHandler(
            entry_points=[
                CallbackQueryHandler(
                    self.callback_auto_mgmt_prompt,
                    pattern='^settings_auto_mgmt_prompt$'),
            ],
            states={
                AUTO_MGMT_DISCLAIMER_SHOWN: [
                    CallbackQueryHandler(
                        self.callback_auto_mgmt_enable,
                        pattern='^auto_mgmt_enable$'),
                    CallbackQueryHandler(
                        self.callback_auto_mgmt_cancel,
                        pattern='^auto_mgmt_cancel$'),
                ],
            },
            fallbacks=[CommandHandler('cancel', self.cmd_cancel)],
            per_user=True,
            per_chat=True,
            per_message=False,
        )
        app.add_handler(auto_mgmt_conv)
        app.add_handler(CallbackQueryHandler(
            self.callback_auto_mgmt_disable, pattern='^auto_mgmt_disable$'
        ))
        app.add_handler(CommandHandler('test_briefing', self.cmd_test_briefing))
        app.add_handler(CommandHandler('test_news',     self.cmd_test_news))
        app.add_handler(CommandHandler('test_weekly',   self.cmd_test_weekly))
        app.add_handler(CommandHandler('test_scan',     self.cmd_test_scan))
        app.add_handler(CommandHandler('upgrade',       self.cmd_upgrade))
        app.add_handler(CallbackQueryHandler(
            self.callback_upgrade_plan, pattern=r'^upgrade_(basic|pro)_(paystack|stripe|bybit)$'
        ))

        self.logger.info("All handlers registered.")

    # ==================== ERROR HANDLER ====================

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """
        Global error handler. Catches all unhandled exceptions including
        network errors (like the ConnectError in the logs).

        Network errors are logged as warnings (they are transient).
        All other errors are logged as errors with the full traceback.
        """
        error = context.error

        if isinstance(error, NetworkError):
            self.logger.warning(
                "Telegram network error (transient, will retry automatically): %s", error
            )
            return

        if isinstance(error, Forbidden):
            # User blocked the bot - nothing we can do
            self.logger.info("User blocked the bot - skipping.")
            return

        self.logger.error(
            "Unhandled error in bot update handler: %s",
            error,
            exc_info=error,
        )

    # ==================== HELPERS ====================

    def _check_rate_limit(self, update: Update) -> bool:
        """Return True if the user is within rate limits."""
        if update.effective_user:
            return _rate_limiter.is_allowed(update.effective_user.id)
        return True

    @staticmethod
    def _ensure_user(telegram_id: int, update: Update) -> dict:
        user   = update.effective_user
        locale = user.language_code if user else None
        tz     = utils.detect_timezone(locale) if hasattr(utils, 'detect_timezone') else 'UTC'
        return db.get_or_create_user(
            telegram_id=telegram_id,
            username=user.username if user else None,
            first_name=user.first_name if user else None,
            user_timezone=tz,
        )

    @staticmethod
    def _format(template: str, **kwargs) -> str:
        """Fill template placeholders and run compliance filter."""
        defaults = {
            'product_name':    config.PRODUCT_NAME,
            'support_contact': config.SUPPORT_CONTACT,
            'footer':          config.FOOTER,
        }
        defaults.update(kwargs)
        filled = template.format(**defaults)
        return utils.validate_user_message(filled)

    async def _reply(self, update: Update, text: str, **kwargs):
        """Reply, splitting at 4000 chars if needed."""
        chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
        for chunk in chunks:
            await update.effective_message.reply_text(chunk, **kwargs)

    async def _send_with_retry(self, chat_id: int, text: str) -> bool:
        """
        Send a Telegram message with exponential backoff retry.
        4 attempts: immediately, then 1s, 2s, 4s delays.
        Returns True if sent successfully.
        """
        for attempt in range(_SEND_MAX_ATTEMPTS):
            try:
                await self.application.bot.send_message(chat_id=chat_id, text=text)
                return True
            except (Forbidden, BadRequest):
                # User blocked the bot or invalid chat - do not retry
                self.logger.info(
                    "Cannot send to user %d: user blocked or chat invalid.", chat_id
                )
                return False
            except NetworkError as err:
                if attempt < _SEND_MAX_ATTEMPTS - 1:
                    delay = _SEND_BASE_DELAY * (2 ** attempt)
                    self.logger.warning(
                        "Network error sending to user %d (attempt %d/%d), "
                        "retrying in %.0f seconds: %s",
                        chat_id, attempt + 1, _SEND_MAX_ATTEMPTS, delay, err,
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        "Failed to send to user %d after %d attempts: %s",
                        chat_id, _SEND_MAX_ATTEMPTS, err,
                    )
            except Exception as err:
                self.logger.error("Unexpected error sending to user %d: %s", chat_id, err)
                return False
        return False

    async def _deliver_queued_messages(self, telegram_id: int):
        """Deliver any messages queued while the user was offline."""
        try:
            pending = db.get_pending_messages(telegram_id)
            if not pending:
                return
            await self._send_with_retry(
                telegram_id,
                f"You have {len(pending)} message(s) that arrived while you were offline:",
            )
            for msg in pending:
                await asyncio.sleep(0.4)  # Avoid hitting Telegram flood limit
                await self._send_with_retry(telegram_id, msg['message_text'])
            db.mark_messages_delivered(telegram_id)
        except Exception as e:
            self.logger.error(
                "Error delivering queued messages to user %d: %s", telegram_id, e
            )

    # ==================== COMMAND HANDLERS ====================

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start."""
        if not self._check_rate_limit(update):
            await self._reply(update, "You are sending commands too quickly. Please wait a moment.")
            return
        try:
            telegram_id = update.effective_user.id
            self._ensure_user(telegram_id, update)
            await self._reply(update, self._format(config.WELCOME_MESSAGE))

            # Generate and send sample chart image (cached after first call)
            global _SAMPLE_CHART_BYTES
            if _SAMPLE_CHART_BYTES is None:
                try:
                    loop = asyncio.get_running_loop()
                    _SAMPLE_CHART_BYTES = await loop.run_in_executor(
                        None, _ChartGenerator.generate_sample_chart)
                except Exception as _sc_err:
                    self.logger.debug("Sample chart generation skipped: %s", _sc_err)

            if _SAMPLE_CHART_BYTES:
                try:
                    from telegram import InputFile as _InputFile
                    await update.effective_message.reply_photo(
                        photo=_InputFile(
                            io.BytesIO(_SAMPLE_CHART_BYTES),
                            filename='sample_setup.png',
                        ),
                        caption=(
                            "Sample chart markup — EURUSD LONG\n"
                            "Order Block (amber), Entry (blue), "
                            "Stop Loss (red), TP1 and TP2 (green).\n"
                            "This is what every setup alert will include."
                        ),
                    )
                except Exception as _photo_err:
                    self.logger.debug(
                        "Sample chart photo send failed: %s", _photo_err)

            await self._reply(
                update,
                "Here is a sample of what the text alert looks like:\n\n"
                + SAMPLE_SETUP_ALERT
            )
            await self._deliver_queued_messages(telegram_id)
        except Exception as e:
            self.logger.error("Error in /start for user %d: %s", update.effective_user.id, e)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help - includes sample alert so users know what to expect."""
        if not self._check_rate_limit(update):
            await self._reply(update, "You are sending commands too quickly. Please wait a moment.")
            return
        try:
            telegram_id = update.effective_user.id
            # Send sample chart so users immediately see what alerts look like
            global _SAMPLE_CHART_BYTES
            if _SAMPLE_CHART_BYTES is None:
                try:
                    loop = asyncio.get_running_loop()
                    _SAMPLE_CHART_BYTES = await loop.run_in_executor(
                        None, _ChartGenerator.generate_sample_chart)
                except Exception as _sc_err:
                    self.logger.debug("Sample chart skipped in /help: %s", _sc_err)

            if _SAMPLE_CHART_BYTES:
                try:
                    from telegram import InputFile as _InputFile
                    await update.effective_message.reply_photo(
                        photo=_InputFile(
                            io.BytesIO(_SAMPLE_CHART_BYTES),
                            filename='sample_setup.png',
                        ),
                        caption=(
                            "Sample setup chart — EURUSD LONG\n"
                            "Basic and Pro subscribers receive a chart "
                            "like this with every setup alert."
                        ),
                    )
                except Exception as _photo_err:
                    self.logger.debug(
                        "Sample chart /help send failed: %s", _photo_err)

            await self._reply(update, self._format(config.HELP_MESSAGE))
            await self._deliver_queued_messages(telegram_id)
        except Exception as e:
            self.logger.error("Error in /help: %s", e)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])

    async def cmd_subscribe(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle /subscribe - show disclaimer first, enforce acceptance."""
        if not self._check_rate_limit(update):
            await self._reply(update, "You are sending commands too quickly. Please wait a moment.")
            return ConversationHandler.END
        try:
            telegram_id = update.effective_user.id
            user        = self._ensure_user(telegram_id, update)

            if user and user.get('subscription_status') == 'active':
                await self._reply(update, self._format(config.ALREADY_SUBSCRIBED))
                return ConversationHandler.END

            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton(
                    "I Understand and Accept", callback_data='disclaimer_accept'
                ),
                InlineKeyboardButton("Decline", callback_data='disclaimer_decline'),
            ]])
            await self._reply(
                update,
                utils.validate_user_message(config.LEGAL_DISCLAIMER),
                reply_markup=keyboard,
            )
            return DISCLAIMER_SHOWN

        except Exception as e:
            self.logger.error("Error in /subscribe: %s", e)
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
            db.accept_disclaimer(telegram_id)
            db.update_subscription(telegram_id, 'active')
            await query.edit_message_text(self._format(config.SUBSCRIBE_SUCCESS))
        except Exception as e:
            self.logger.error("Error in disclaimer_accept: %s", e)
            await query.edit_message_text(config.ERROR_MESSAGES['general_error'])
        return ConversationHandler.END

    async def callback_disclaimer_decline(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        query = update.callback_query
        await query.answer()
        await query.edit_message_text(
            utils.validate_user_message(
                "Subscription not activated.\n\n"
                "You must accept the disclaimer to receive automated setup alerts.\n\n"
                f"{config.FOOTER}"
            )
        )
        return ConversationHandler.END

    async def cmd_connect_mt5(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle /connect_mt5 - collect credentials one field at a time."""
        if not self._check_rate_limit(update):
            await self._reply(update, "You are sending commands too quickly. Please wait a moment.")
            return ConversationHandler.END
        try:
            telegram_id = update.effective_user.id
            user        = db.get_user(telegram_id)

            # If already connected, show current account and offer to re-connect
            if user and user.get('mt5_connected'):
                masked = '****%s' % str(user.get('mt5_login', ''))[-4:]
                broker = user.get('mt5_broker_name', 'N/A')
                keyboard = InlineKeyboardMarkup([[
                    InlineKeyboardButton(
                        "Re-connect with new account", callback_data='mt5_reconnect_confirm'),
                    InlineKeyboardButton("Cancel", callback_data='mt5_reconnect_cancel'),
                ]])
                await self._reply(
                    update,
                    utils.validate_user_message(
                        "Your trading account is already connected.\n\n"
                        "Account: %s\n"
                        "Broker:  %s\n\n"
                        "Do you want to disconnect and connect a different account?\n\n"
                        "%s" % (masked, broker, config.FOOTER)
                    ),
                    reply_markup=keyboard,
                )
                return ConversationHandler.END

            context.user_data.clear()
            await self._reply(
                update,
                utils.validate_user_message(
                    "CONNECT YOUR MT5 ACCOUNT\n\n"
                    "I will ask for 3 things one at a time:\n"
                    "  1. Account number\n"
                    "  2. Password\n"
                    "  3. Broker server name\n\n"
                    "STEP 1 of 3\n\n"
                    "Please type your MT5 account NUMBER.\n"
                    "This is a 5 to 12 digit number (e.g. 12345678).\n"
                    "You can find it in MT5: File - Login to Trade Account.\n\n"
                    "Send /cancel at any time to stop without saving.\n\n"
                    "%s" % config.FOOTER
                ),
            )
            return MT5_WAITING_LOGIN
        except Exception as e:
            self.logger.error("Error in /connect_mt5: %s", e)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])
            return ConversationHandler.END

    async def handle_mt5_login(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Receive and validate MT5 account number."""
        raw         = (update.message.text or '').strip()
        telegram_id = update.effective_user.id
        is_group    = update.message.chat.type != 'private'

        # Delete message in group chats to protect the account number
        try:
            await update.message.delete()
        except Exception:
            # Cannot delete in private chats - instruct user manually
            try:
                await update.message.reply_text(
                    "For your security, please delete the message containing your account number."
                )
            except Exception:
                pass

        if not raw.isdigit() or not (5 <= len(raw) <= 12):
            await self._reply(
                update,
                utils.validate_user_message(
                    "That does not look like a valid account number.\n"
                    "Please enter digits only, 5 to 12 characters long.\n\n"
                    f"{config.FOOTER}"
                ),
            )
            return MT5_WAITING_LOGIN

        context.user_data['mt5_login'] = int(raw)
        await self._reply(
            update,
            utils.validate_user_message(
                "Account number received.\n\n"
                "Now please type your MT5 PASSWORD:\n\n"
                f"{config.FOOTER}"
            ),
        )
        return MT5_WAITING_PASSWORD

    async def handle_mt5_password(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Receive MT5 password. Always attempt to delete the message."""
        raw         = (update.message.text or '').strip()
        is_group    = update.message.chat.type != 'private'

        # Always try to delete password messages everywhere
        try:
            await update.message.delete()
        except Exception:
            # Cannot delete in private chats - instruct user
            if not is_group:
                try:
                    await update.message.reply_text(
                        "For your security, please delete the message containing your password."
                    )
                except Exception:
                    pass

        if not raw or len(raw) < 4:
            await self._reply(
                update,
                utils.validate_user_message(
                    "Password seems too short. Please try again.\n\n"
                    f"{config.FOOTER}"
                ),
            )
            return MT5_WAITING_PASSWORD

        context.user_data['mt5_password'] = raw
        await self._reply(
            update,
            utils.validate_user_message(
                "Password received.\n\n"
                "Finally, please type your BROKER SERVER name.\n"
                "Examples: ICMarkets-Live01  |  Exness-Real  |  XM-Real\n\n"
                "You can find this in MT5: Tools - Options - Server tab.\n\n"
                f"{config.FOOTER}"
            ),
        )
        return MT5_WAITING_SERVER

    async def handle_mt5_server(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Receive server name, verify with worker, save if valid."""
        raw         = (update.message.text or '').strip()
        telegram_id = update.effective_user.id
        is_group    = update.message.chat.type != 'private'

        if is_group:
            try:
                await update.message.delete()
            except Exception:
                pass

        if not raw or len(raw) < 3:
            await self._reply(
                update,
                utils.validate_user_message(
                    "Server name is too short.\n\n"
                    "The server name looks like: Exness-MT5Real4 or ICMarkets-Live01\n"
                    "You can find it in MT5: Tools - Options - Server tab.\n\n"
                    "Please type the server name again, or send /cancel to stop.\n\n"
                    "%s" % config.FOOTER
                ),
            )
            return MT5_WAITING_SERVER

        login    = context.user_data.get('mt5_login')
        password = context.user_data.get('mt5_password')
        context.user_data.clear()  # Wipe credentials from memory immediately

        await self._reply(
            update,
            utils.validate_user_message(
                "Checking your account details with your broker. This may take up to 15 seconds...\n\n"
                f"{config.FOOTER}"
            ),
        )

        try:
            success, result = await self.mt5.verify_credentials(telegram_id, login, password, raw)

            if success and isinstance(result, dict):
                # ---- Account limit check ----
                sub_mgr   = get_subscription_manager()
                can_add, limit_reason = sub_mgr.can_add_account(telegram_id)
                if not can_add:
                    await self._reply(
                        update,
                        utils.validate_user_message(
                            "Account connection blocked.\n\n"
                            f"{limit_reason}\n\n"
                            f"{config.FOOTER}"
                        ),
                    )
                    return ConversationHandler.END

                # ---- Foreign currency check (Pro and Admin only) ----
                account_currency = result.get('currency', 'USD')
                if account_currency != 'USD' and not sub_mgr.can_use_foreign_currency(telegram_id):
                    await self._reply(
                        update,
                        utils.validate_user_message(
                            f"Non-USD accounts ({account_currency}) require a Pro subscription.\n\n"
                            "Upgrade to Pro ($100/month) to connect accounts in any currency.\n"
                            "Use /upgrade to continue.\n\n"
                            f"{config.FOOTER}"
                        ),
                    )
                    return ConversationHandler.END

                db.save_mt5_credentials(
                    telegram_id=telegram_id,
                    login=login,
                    password=password,
                    server=raw,
                    broker_name=result.get('broker', ''),
                    balance=result.get('balance', 0.0),
                    currency=account_currency,
                    metaapi_account_id=result.get('metaapi_account_id', ''),
                )
                masked = f"****{str(login)[-4:]}"
                trade_status = (
                    "Automated execution is now active."
                    if result.get('trade_allowed')
                    else
                    "Connected. Note: Automated trading is not enabled on this account. "
                    "Enable it in MT5: Tools - Options - Expert Advisors - Allow automated trading."
                )
                await self._reply(
                    update,
                    utils.validate_user_message(
                        f"Account connected successfully.\n\n"
                        f"Account: {masked}\n"
                        f"Broker:  {result.get('broker', 'N/A')}\n"
                        f"Balance: {result.get('currency', 'USD')} "
                        f"{result.get('balance', 0.0):,.2f}\n"
                        f"Server:  {raw}\n\n"
                        f"{trade_status}\n\n"
                        f"{config.FOOTER}"
                    ),
                )
            else:
                err = result if isinstance(result, str) else "Verification failed."
                await self._reply(
                    update,
                    utils.validate_user_message(
                        f"Connection failed.\n\nReason: {err}\n\n"
                        "Please check:\n"
                        "1. Account number is correct (digits only)\n"
                        "2. Password is correct (passwords are case-sensitive)\n"
                        "3. Server name matches exactly what appears in MT5\n\n"
                        "Use /connect_mt5 to try again.\n\n"
                        f"{config.FOOTER}"
                    ),
                )

        except Exception as e:
            self.logger.error("Error verifying MT5 credentials for user %d: %s", telegram_id, e)
            await self._reply(update, config.ERROR_MESSAGES['service_unavailable'])

        return ConversationHandler.END

    async def cmd_disconnect_mt5(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /disconnect_mt5."""
        if not self._check_rate_limit(update):
            await self._reply(update, "You are sending commands too quickly. Please wait a moment.")
            return
        try:
            telegram_id = update.effective_user.id
            user        = db.get_user(telegram_id)
            if not user or not user.get('mt5_connected'):
                await self._reply(update, config.ERROR_MESSAGES['mt5_not_connected'])
                return
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("Yes, Disconnect", callback_data='mt5_disconnect_confirm'),
                InlineKeyboardButton("Cancel",          callback_data='mt5_disconnect_cancel'),
            ]])
            await self._reply(
                update,
                utils.validate_user_message(
                    "Are you sure you want to disconnect your trading account?\n\n"
                    "Automated trade execution will stop immediately.\n\n"
                    f"{config.FOOTER}"
                ),
                reply_markup=keyboard,
            )
        except Exception as e:
            self.logger.error("Error in /disconnect_mt5: %s", e)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])

    async def callback_disconnect_confirm(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        query = update.callback_query
        await query.answer()
        try:
            db.delete_mt5_credentials(update.effective_user.id)
            await query.edit_message_text(
                utils.validate_user_message(
                    "Trading account disconnected.\n\n"
                    "You will still receive setup alerts but no trades will be placed automatically.\n\n"
                    f"{config.FOOTER}"
                )
            )
        except Exception as e:
            self.logger.error("Error in disconnect_confirm: %s", e)
            await query.edit_message_text(config.ERROR_MESSAGES['general_error'])

    async def callback_disconnect_cancel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        query = update.callback_query
        await query.answer()
        await query.edit_message_text(
            utils.validate_user_message(f"Disconnection cancelled.\n\n{config.FOOTER}")
        )
        
    async def callback_mt5_reconnect_confirm(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """User confirmed they want to re-connect with a new account."""
        query = update.callback_query
        await query.answer()
        telegram_id = query.from_user.id
        db.delete_mt5_credentials(telegram_id)
        await query.edit_message_text(
            utils.validate_user_message(
                "Previous account disconnected.\n\n"
                "Use /connect_mt5 to connect your new account.\n\n"
                "%s" % config.FOOTER
            )
        )

    async def callback_mt5_reconnect_cancel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        await query.answer()
        await query.edit_message_text(
            utils.validate_user_message(
                "No changes made. Your account remains connected.\n\n%s" % config.FOOTER
            )
        )

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status."""
        if not self._check_rate_limit(update):
            await self._reply(update, "You are sending commands too quickly. Please wait a moment.")
            return
        try:
            telegram_id = update.effective_user.id
            user        = db.get_user(telegram_id)
            if not user:
                await self._reply(update, "Please use /start first.")
                return

            sub_status = user.get('subscription_status', 'inactive').upper()
            mt5_ok     = bool(user.get('mt5_connected'))
            risk_pct   = user.get('risk_percent', config.DEFAULT_RISK_PERCENT)
            stats      = db.get_user_statistics(telegram_id) or {}

            lines = [
                "YOUR ACCOUNT STATUS",
                "",
                f"Subscription:        {sub_status}",
                f"MT5 Connected:       {'YES' if mt5_ok else 'NO'}",
                f"Risk Per Trade:      {risk_pct}%",
            ]
            if mt5_ok:
                lines += [
                    f"Broker:              {user.get('mt5_broker_name', 'N/A')}",
                    f"Account:             ****{str(user.get('mt5_login', ''))[-4:]}",
                ]
            concluded = stats.get('concluded_trades', 0)
            open_ct   = stats.get('open_trades', 0)
            wins      = stats.get('wins', 0)
            losses    = stats.get('losses', 0)
            breakevens = stats.get('breakevens', 0)
            lines += [
                "",
                "TRADING STATISTICS",
                "",
                f"Total Setups Executed: {stats.get('total_trades', 0)}",
                f"Currently Open:        {open_ct}",
                f"Closed Trades:         {stats.get('closed_trades', 0)}",
                f"  Wins:                {wins}",
                f"  Losses:              {losses}",
                f"  Breakeven:           {breakevens}",
                f"Historical Success Rate: {stats.get('win_rate', 0.0):.1f}%"
                f"  (based on {concluded} concluded trade(s))",
                "  Past performance does not guarantee future results.",
                f"Total Pips (Closed):   {stats.get('total_pips', 0.0):+.1f}",
                "",
                "Use /settings to adjust your risk percentage.",
                f"\n{config.FOOTER}",
            ]
            await self._reply(
                update, utils.validate_user_message("\n".join(lines))
            )
            await self._deliver_queued_messages(telegram_id)
        except Exception as e:
            self.logger.error("Error in /status: %s", e)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])

    async def cmd_latest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /latest - show most recent setup or sample if none exists."""
        if not self._check_rate_limit(update):
            await self._reply(update, "You are sending commands too quickly. Please wait a moment.")
            return
        try:
            telegram_id = update.effective_user.id
            user        = db.get_user(telegram_id)

            if not user or user.get('subscription_status') != 'active':
                await self._reply(update, config.ERROR_MESSAGES['not_subscribed'])
                return

            signal = db.get_latest_signal()
            if signal:
                msg = utils.format_setup_message(
                    signal_number=signal.get('signal_number', 0),
                    symbol=signal['symbol'],
                    direction=signal['direction'],
                    setup_type=signal['setup_type'],
                    entry_price=signal['entry_price'],
                    stop_loss=signal['stop_loss'],
                    take_profit_1=signal['take_profit_1'],
                    take_profit_2=signal['take_profit_2'],
                    sl_pips=signal.get('sl_pips', 0.0),
                    tp1_pips=signal.get('tp1_pips', 0.0),
                    tp2_pips=signal.get('tp2_pips', 0.0),
                    rr_tp1=signal.get('rr_tp1', 0.0),
                    rr_tp2=signal.get('rr_tp2', 0.0),
                    ml_score=signal.get('ml_score', 0),
                    session=signal.get('session', 'N/A'),
                    order_type=signal.get('order_type', 'LIMIT'),
                    lot_size=None,
                    expiry_hours=int(signal.get('expiry_hours', 8)),
                )
                await self._reply(update, msg)
            else:
                await self._reply(
                    update,
                    utils.validate_user_message(
                        "No setups have been detected yet.\n\n"
                        "The system scans every 15 minutes. "
                        "A qualifying setup will be sent automatically when conditions align.\n\n"
                        + config.FOOTER
                    )
                )
        except Exception as e:
            self.logger.error("Error in /latest: %s", e)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])

    async def cmd_settings(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """
        Handle /settings.
        Shows risk percentage as inline keyboard buttons.
        Timezone is set by pressing the timezone button (separate conversation entry).
        """
        if not self._check_rate_limit(update):
            await self._reply(update, "You are sending commands too quickly. Please wait a moment.")
            return ConversationHandler.END
        try:
            telegram_id = update.effective_user.id
            user        = db.get_user(telegram_id)

            from payment_handler import get_subscription_manager as _gsm
            if _gsm().get_tier(telegram_id) == 'free':
                await self._reply(
                    update,
                    utils.validate_user_message(
                        "Settings and execution preferences require a Basic "
                        "or higher subscription.\n\n"
                        "Use /upgrade to view plans and unlock risk settings, "
                        "timezone configuration, and automated trade execution.\n\n"
                        f"{config.FOOTER}"
                    )
                )
                return ConversationHandler.END

            current     = (
                user.get('risk_percent', config.DEFAULT_RISK_PERCENT)
                if user else config.DEFAULT_RISK_PERCENT
            )
            tz = (
                user.get('timezone') or user.get('user_timezone') or 'UTC'
            ) if user else 'UTC'

            auto_mgmt_enabled = bool(user.get('auto_position_management', False)) if user else False
            auto_mgmt_label   = (
                "Auto-Manage Positions: ON  (tap to disable)"
                if auto_mgmt_enabled
                else "Auto-Manage Positions: OFF  (tap to enable)"
            )
            auto_mgmt_callback = (
                'auto_mgmt_disable'
                if auto_mgmt_enabled
                else 'settings_auto_mgmt_prompt'
            )

            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton(
                        f"0.5%{'  (current)' if current == 0.5 else ''}",
                        callback_data='risk_0.5'),
                    InlineKeyboardButton(
                        f"1%{'  (current)' if current == 1.0 else ''}",
                        callback_data='risk_1.0'),
                    InlineKeyboardButton(
                        f"1.5%{'  (current)' if current == 1.5 else ''}",
                        callback_data='risk_1.5'),
                ],
                [
                    InlineKeyboardButton(
                        f"2%{'  (current)' if current == 2.0 else ''}",
                        callback_data='risk_2.0'),
                    InlineKeyboardButton(
                        f"3%{'  (current)' if current == 3.0 else ''}",
                        callback_data='risk_3.0'),
                    InlineKeyboardButton(
                        f"5%{'  (current)' if current == 5.0 else ''}",
                        callback_data='risk_5.0'),
                ],
                [
                    InlineKeyboardButton(
                        "Set Timezone",
                        callback_data='settings_tz_prompt'),
                ],
                [
                    InlineKeyboardButton(
                        auto_mgmt_label,
                        callback_data=auto_mgmt_callback),
                ],
                [
                    InlineKeyboardButton("Done", callback_data='settings_done'),
                ],
            ])

            auto_mgmt_status = "ENABLED" if auto_mgmt_enabled else "DISABLED"
            await self._reply(
                update,
                utils.validate_user_message(
                    f"SETTINGS\n\n"
                    f"Current risk per trade:    {current}%\n"
                    f"Current timezone:          {tz}\n"
                    f"Auto position management:  {auto_mgmt_status}\n\n"
                    f"Tap a risk percentage to change it.\n"
                    f"Tap 'Set Timezone' to change your timezone.\n"
                    f"Tap 'Auto-Manage Positions' to control whether the bot\n"
                    f"manages TP1, breakeven, and TP2 automatically without\n"
                    f"asking for your confirmation each time.\n\n"
                    f"Recommended risk levels:\n"
                    f"  Beginner:     0.5%\n"
                    f"  Intermediate: 1% - 2%\n"
                    f"  Advanced:     2% - 3%\n\n"
                    f"{config.FOOTER}"
                ),
                reply_markup=keyboard,
            )
            return ConversationHandler.END

        except Exception as e:
            self.logger.error("Error in /settings: %s", e)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])
            return ConversationHandler.END

    async def callback_settings_risk(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle risk percentage button press from settings keyboard."""
        query = update.callback_query
        await query.answer()
        telegram_id = query.from_user.id
        try:
            # callback_data is like "risk_1.5" - split on underscore, take last part
            risk = float(query.data.split('_')[1])
            if not (config.MIN_RISK_PERCENT <= risk <= config.MAX_RISK_PERCENT):
                await query.edit_message_text(
                    utils.validate_user_message(
                        f"Invalid risk value. Please use /settings again.\n\n{config.FOOTER}"
                    )
                )
                return
            db.update_risk_percent(telegram_id, risk)
            await query.edit_message_text(
                utils.validate_user_message(
                    f"Risk per trade set to {risk}%.\n\n"
                    f"This applies to all future automated trades.\n\n"
                    f"Use /settings to change it again.\n\n"
                    f"{config.FOOTER}"
                )
            )
        except Exception as e:
            self.logger.error("Error in callback_settings_risk: %s", e)
            await query.edit_message_text(config.ERROR_MESSAGES['general_error'])

    async def callback_settings_tz_prompt(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """
        Handle 'Set Timezone' button press.
        Shows an inline keyboard of common timezones.
        Each button press is handled by callback_settings_tz_select.
        """
        query = update.callback_query
        await query.answer()

        _TIMEZONES = [
            ('America/New_York',     'US Eastern'),
            ('America/Chicago',      'US Central'),
            ('America/Los_Angeles',  'US Pacific'),
            ('Europe/London',        'UK'),
            ('Europe/Paris',         'Central Europe'),
            ('Europe/Moscow',        'Moscow'),
            ('Africa/Lagos',         'West Africa'),
            ('Africa/Johannesburg',  'South Africa'),
            ('Asia/Dubai',           'Dubai'),
            ('Asia/Kolkata',         'India'),
            ('Asia/Singapore',       'Singapore'),
            ('Asia/Tokyo',           'Japan'),
            ('Asia/Shanghai',        'China'),
            ('Australia/Sydney',     'Australia East'),
            ('Pacific/Auckland',     'New Zealand'),
        ]

        buttons = []
        for i in range(0, len(_TIMEZONES), 3):
            row = []
            for tz_iana, tz_label in _TIMEZONES[i:i + 3]:
                safe = tz_iana.replace('/', '~')
                row.append(InlineKeyboardButton(
                    tz_label,
                    callback_data=f'tz_sel_{safe}',
                ))
            buttons.append(row)

        keyboard = InlineKeyboardMarkup(buttons)

        await query.edit_message_text(
            utils.validate_user_message(
                "SELECT YOUR TIMEZONE\n\n"
                "Choose your region from the buttons below.\n"
                "Daily briefings and news alerts will arrive at the correct local time.\n\n"
                f"{config.FOOTER}"
            ),
            reply_markup=keyboard,
        )
        return ConversationHandler.END
    
    
    async def callback_settings_tz_select(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle timezone selection from the inline keyboard shown by callback_settings_tz_prompt."""
        query = update.callback_query
        await query.answer()
        telegram_id = query.from_user.id

        raw_safe = query.data.replace('tz_sel_', '')
        tz_str   = raw_safe.replace('~', '/')

        try:
            import pytz
            pytz.timezone(tz_str)
            db.update_timezone(telegram_id, tz_str)
            await query.edit_message_text(
                utils.validate_user_message(
                    f"Timezone updated to {tz_str}.\n\n"
                    "Daily briefings and news alerts will now arrive at the correct time.\n\n"
                    f"{config.FOOTER}"
                )
            )
        except Exception as e:
            self.logger.error("Timezone selection error for user %d: %s", telegram_id, e)
            await query.edit_message_text(
                utils.validate_user_message(
                    "That timezone could not be saved. "
                    "Please use /settings and try again.\n\n"
                    f"{config.FOOTER}"
                )
            )

    async def handle_settings_risk(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Receive new risk percentage."""
        raw         = (update.message.text or '').strip().replace('%', '')
        telegram_id = update.effective_user.id
        try:
            risk = float(raw)
        except ValueError:
            await self._reply(
                update,
                utils.validate_user_message(
                    "That is not a valid number. Please enter a number like 1.0 or 2.5.\n\n"
                    f"{config.FOOTER}"
                ),
            )
            return SETTINGS_WAITING_RISK

        if not (config.MIN_RISK_PERCENT <= risk <= config.MAX_RISK_PERCENT):
            await self._reply(
                update,
                utils.validate_user_message(
                    f"Risk must be between {config.MIN_RISK_PERCENT}% "
                    f"and {config.MAX_RISK_PERCENT}%.\n\n"
                    f"{config.FOOTER}"
                ),
            )
            return SETTINGS_WAITING_RISK

        db.update_risk_percent(telegram_id, risk)
        await self._reply(
            update,
            utils.validate_user_message(
                f"Risk per trade updated to {risk}%.\n\n"
                "This will apply to all future automated trades.\n\n"
                f"{config.FOOTER}"
            ),
        )
        return ConversationHandler.END

    async def handle_settings_tz(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Receive timezone setting."""
        raw         = (update.message.text or '').strip()
        telegram_id = update.effective_user.id
        try:
            import pytz
            pytz.timezone(raw)
        except Exception:
            await self._reply(
                update,
                utils.validate_user_message(
                    f"'{raw}' is not a recognised timezone.\n"
                    "Use an IANA timezone name like: America/New_York or Europe/London\n\n"
                    f"{config.FOOTER}"
                ),
            )
            return SETTINGS_WAITING_TZ

        db.update_timezone(telegram_id, raw)
        await self._reply(
            update,
            utils.validate_user_message(
                f"Timezone updated to {raw}.\n\n"
                "Daily market briefings will now arrive at 8:00 AM in this timezone.\n\n"
                f"{config.FOOTER}"
            ),
        )
        return ConversationHandler.END

    async def callback_settings_done(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        query = update.callback_query
        await query.answer()
        await query.edit_message_text(
            utils.validate_user_message(f"Settings saved.\n\n{config.FOOTER}")
        )
        return ConversationHandler.END
    
    async def callback_auto_mgmt_prompt(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """
        User tapped 'Auto-Manage Positions: OFF' in settings.
        Show the autonomous management disclaimer with Accept/Cancel buttons.
        User must explicitly accept before the feature is enabled.
        """
        query = update.callback_query
        await query.answer()
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton(
                "Enable Autonomous Management",
                callback_data='auto_mgmt_enable'),
            InlineKeyboardButton(
                "Cancel",
                callback_data='auto_mgmt_cancel'),
        ]])
        await query.edit_message_text(
            utils.validate_user_message(AUTO_MGMT_DISCLAIMER),
            reply_markup=keyboard,
        )
        return AUTO_MGMT_DISCLAIMER_SHOWN

    async def callback_auto_mgmt_enable(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """User accepted the autonomous management disclaimer. Enable the feature."""
        query = update.callback_query
        await query.answer()
        telegram_id = query.from_user.id
        try:
            db.update_auto_position_management(telegram_id, True)
            await query.edit_message_text(
                utils.validate_user_message(
                    "Autonomous position management is now ENABLED.\n\n"
                    "The bot will automatically close 50 percent of your position "
                    "at TP1, move the stop loss to breakeven, and close the remaining "
                    "50 percent at TP2 without asking for confirmation.\n\n"
                    "You will still receive a notification after each action is taken.\n\n"
                    "To disable this, use /settings at any time.\n\n"
                    f"{config.FOOTER}"
                )
            )
        except Exception as e:
            self.logger.error(
                "Error enabling auto position management for user %d: %s",
                telegram_id, e)
            await query.edit_message_text(config.ERROR_MESSAGES['general_error'])
        return ConversationHandler.END

    async def callback_auto_mgmt_cancel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """User cancelled the autonomous management disclaimer. No change."""
        query = update.callback_query
        await query.answer()
        await query.edit_message_text(
            utils.validate_user_message(
                "No changes made. Autonomous position management remains disabled.\n\n"
                "The bot will continue to ask for your confirmation at TP1.\n\n"
                f"{config.FOOTER}"
            )
        )
        return ConversationHandler.END

    async def callback_auto_mgmt_disable(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """User tapped 'Auto-Manage Positions: ON' to disable the feature."""
        query = update.callback_query
        await query.answer()
        telegram_id = query.from_user.id
        try:
            db.update_auto_position_management(telegram_id, False)
            await query.edit_message_text(
                utils.validate_user_message(
                    "Autonomous position management is now DISABLED.\n\n"
                    "The bot will ask for your confirmation before taking "
                    "partial profit at TP1.\n\n"
                    "Use /settings to enable it again at any time.\n\n"
                    f"{config.FOOTER}"
                )
            )
        except Exception as e:
            self.logger.error(
                "Error disabling auto position management for user %d: %s",
                telegram_id, e)
            await query.edit_message_text(config.ERROR_MESSAGES['general_error'])

    async def cmd_unsubscribe(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /unsubscribe with confirmation."""
        if not self._check_rate_limit(update):
            await self._reply(update, "You are sending commands too quickly. Please wait a moment.")
            return
        try:
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("Yes, Unsubscribe", callback_data='unsub_confirm'),
                InlineKeyboardButton("Cancel",           callback_data='unsub_cancel'),
            ]])
            await self._reply(
                update,
                utils.validate_user_message(
                    "Are you sure you want to unsubscribe?\n\n"
                    "You will stop receiving automated setup alerts.\n"
                    "You can resubscribe at any time with /subscribe.\n\n"
                    f"{config.FOOTER}"
                ),
                reply_markup=keyboard,
            )
        except Exception as e:
            self.logger.error("Error in /unsubscribe: %s", e)
            await self._reply(update, config.ERROR_MESSAGES['general_error'])

    async def callback_unsubscribe_confirm(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        query = update.callback_query
        await query.answer()
        try:
            db.update_subscription(update.effective_user.id, 'inactive')
            await query.edit_message_text(
                utils.validate_user_message(
                    "You have been unsubscribed.\n\n"
                    "You will no longer receive automated setup alerts.\n"
                    "Use /subscribe to reactivate at any time.\n\n"
                    f"{config.FOOTER}"
                )
            )
        except Exception as e:
            self.logger.error("Error in unsubscribe_confirm: %s", e)
            await query.edit_message_text(config.ERROR_MESSAGES['general_error'])

    async def callback_unsubscribe_cancel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        query = update.callback_query
        await query.answer()
        await query.edit_message_text(
            utils.validate_user_message(f"Unsubscription cancelled.\n\n{config.FOOTER}")
        )

    async def cmd_cancel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Universal cancel for any active conversation."""
        context.user_data.clear()
        await self._reply(
            update,
            utils.validate_user_message(
                "Action cancelled.\n\nUse /help to see available commands.\n\n"
                f"{config.FOOTER}"
            ),
        )
        return ConversationHandler.END
    
    
    async def callback_partial_close(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle partial profit confirmation from TP1 inline keyboard."""
        query = update.callback_query
        await query.answer()
        telegram_id = query.from_user.id
        data        = query.data

        try:
            parts  = data.split('_')
            action = parts[2]
            ticket = int(parts[3])
        except (IndexError, ValueError):
            await query.edit_message_text("Invalid response. Position unchanged.")
            return

        position = None
        if self.position_monitor:
            position = self.position_monitor.get_position_by_ticket(ticket)

        if position is None:
            await query.edit_message_text(
                "Position %d not found. It may have already been closed." % ticket)
            return

        if not getattr(position, 'awaiting_partial_confirm', False):
            await query.edit_message_text(
                "This confirmation has already been processed or has expired.")
            return

        position.awaiting_partial_confirm = False

        if action == 'yes':
            half_lots = round(position.volume / 2, 2)
            success, msg = await self.mt5.close_partial_position(
                telegram_id=telegram_id,
                ticket=ticket,
                close_pct=0.5,
            )
            if success:
                await query.edit_message_text(
                    utils.validate_user_message(
                        "PARTIAL CLOSE CONFIRMED\n\n"
                        "Symbol:    %s\n"
                        "Ticket:    %d\n"
                        "Closed:    %.2f lots\n\n"
                        "Remaining position is running to TP2.\n"
                        "Stop loss is being moved to breakeven.\n\n"
                        "EDUCATIONAL PURPOSES ONLY. NOT FINANCIAL ADVICE.\n\n"
                        "%s" % (
                            position.symbol, ticket, half_lots, config.FOOTER
                        )
                    )
                )
                asyncio.create_task(
                    self._move_position_sl_to_breakeven(telegram_id, position))
            else:
                await query.edit_message_text(
                    utils.validate_user_message(
                        "PARTIAL CLOSE FAILED\n\n"
                        "Symbol:    %s\n"
                        "Ticket:    %d\n\n"
                        "Reason: %s\n\n"
                        "Your full position remains open and is being "
                        "monitored automatically.\n\n"
                        "EDUCATIONAL PURPOSES ONLY. NOT FINANCIAL ADVICE.\n\n"
                        "%s" % (
                            position.symbol, ticket, msg, config.FOOTER
                        )
                    )
                )
        else:
            # User declined partial close.
            # Ask whether they want breakeven protection in the same message style.
            be_keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton(
                    "Yes - Move to Breakeven",
                    callback_data="breakeven_yes_%d" % ticket,
                ),
                InlineKeyboardButton(
                    "No - Keep Current Stop",
                    callback_data="breakeven_no_%d" % ticket,
                ),
            ]])
            await query.edit_message_text(
                utils.validate_user_message(
                    "BREAKEVEN PROTECTION - YOUR DECISION REQUIRED\n\n"
                    "Symbol:     %s\n"
                    "Direction:  %s\n"
                    "Ticket:     %d\n\n"
                    "You chose to hold the full position to TP2.\n\n"
                    "Option 1: Move stop loss to entry price plus 5 pip buffer "
                    "to protect from a loss if price reverses.\n\n"
                    "Option 2: Keep the current stop loss in place and "
                    "hold the full position as it stands.\n\n"
                    "EDUCATIONAL PURPOSES ONLY. NOT FINANCIAL ADVICE." % (
                        position.symbol,
                        position.direction,
                        ticket,
                    )
                ),
                reply_markup=be_keyboard,
            )

    async def _move_position_sl_to_breakeven(
        self, telegram_id: int, position
    ) -> None:
        """Move stop loss to entry price after user confirms partial close at TP1."""
        try:
            pip_value = utils.get_pip_value(position.symbol)
            buffer    = getattr(config, 'BREAKEVEN_BUFFER_PIPS', 2) * pip_value
            be_price  = (
                position.entry_price + buffer
                if position.direction == 'BUY'
                else position.entry_price - buffer
            )
            success, msg = await self.mt5.modify_stop_loss(
                telegram_id=telegram_id,
                ticket=position.ticket,
                new_sl=be_price,
            )
            if success:
                position.be_activated = True
                position.stop_loss    = be_price
            else:
                self.logger.warning(
                    "Could not move SL to breakeven for ticket %d: %s",
                    position.ticket, msg)
        except Exception as e:
            self.logger.error(
                "Error moving SL to breakeven for ticket %d: %s",
                position.ticket, e)
            
    async def callback_breakeven_decision(
        self, update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle the inline keyboard response to the breakeven question.
        Callback data format: 'breakeven_yes_TICKET' or 'breakeven_no_TICKET'.
        Shown to the user after they decline partial close at TP1.
        """
        query = update.callback_query
        await query.answer()
        telegram_id = query.from_user.id
        data        = query.data

        try:
            parts  = data.split('_')
            action = parts[1]        # 'yes' or 'no'
            ticket = int(parts[2])
        except (IndexError, ValueError):
            await query.edit_message_text(
                "Invalid response. Position has not been changed."
            )
            return

        position = None
        if self.position_monitor:
            position = self.position_monitor.get_position_by_ticket(ticket)

        if position is None:
            await query.edit_message_text(
                utils.validate_user_message(
                    "Position %d was not found. "
                    "It may have already been closed automatically.\n\n"
                    "%s" % (ticket, config.FOOTER)
                )
            )
            return

        if action == 'yes':
            asyncio.create_task(
                self._move_position_sl_to_breakeven(telegram_id, position)
            )
            await query.edit_message_text(
                utils.validate_user_message(
                    "Stop loss is being moved to breakeven for %s #%d.\n\n"
                    "Your full position continues to TP2 automatically.\n\n"
                    "%s" % (position.symbol, ticket, config.FOOTER)
                )
            )
        else:
            # User explicitly chose to keep the current stop loss.
            # Set the flag so the position monitor does not automatically
            # activate breakeven on the next 10-second cycle, which would
            # override the user's decision.
            position.user_declined_breakeven = True
            await query.edit_message_text(
                utils.validate_user_message(
                    "Understood. No changes made for %s #%d.\n\n"
                    "Your full position continues to TP2 at the current stop loss.\n\n"
                    "%s" % (position.symbol, ticket, config.FOOTER)
                )
            )


    async def cmd_download(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        /download - Generate and send two CSV files to the user:
          1. nixie_trading_history_TIMESTAMP.csv  - all their trades
          2. nixie_ml_setups_TIMESTAMP.csv        - all automated setups with ML scores

        Only subscribed users can download.
        Files are sent as Telegram document attachments.
        """
        import csv
        import io
        from telegram import InputFile

        telegram_id = update.effective_user.id
        self.logger.info("User %d requested /download", telegram_id)

        user = db.get_user(telegram_id)
        if not user or user.get('subscription_status') != 'active':
            await self._reply(
                update,
                "This feature requires an active subscription.\n\n"
                "Use /subscribe to get started.\n\n%s" % config.FOOTER,
            )
            return

        from payment_handler import get_subscription_manager as _gsm
        _tier = _gsm().get_tier(telegram_id)
        if _tier == 'free':
            await self._reply(
                update,
                utils.validate_user_message(
                    "Downloading your trading history requires a Basic or "
                    "higher subscription.\n\n"
                    "Use /upgrade to unlock trading records, performance "
                    "analytics, and CSV exports.\n\n"
                    f"{config.FOOTER}"
                )
            )
            return

        is_admin = telegram_id in config.ADMIN_USER_IDS

        await self._reply(update, "Generating your CSV files. Please wait...")

        try:
            now_str = utils.get_current_utc_time().strftime('%Y%m%d_%H%M%S')

            # ---- 1. Trading history CSV ----
            trades = db.get_all_trades_for_csv(telegram_id)
            trade_buf = io.StringIO()
            w = csv.writer(trade_buf)
            if trades:
                w.writerow(list(trades[0].keys()))
                for row in trades:
                    w.writerow(list(row.values()))
            else:
                w.writerow(['telegram_id', 'symbol', 'direction', 'lot_size',
                            'entry_price', 'stop_loss', 'take_profit_1', 'take_profit_2',
                            'order_type', 'status', 'realized_pnl', 'rr_achieved',
                            'opened_at', 'closed_at'])
                w.writerow(['No trades yet.'])

            await context.bot.send_document(
                chat_id=telegram_id,
                document=InputFile(
                    io.BytesIO(trade_buf.getvalue().encode('utf-8')),
                    filename=f"nixie_trading_history_{now_str}.csv"
                ),
                caption=(
                    f"Your trading history ({len(trades)} trade records).\n"
                    f"Generated: {now_str} UTC"
                ),
            )

            # ---- 2. ML setups CSV (admin only) ----
            if is_admin:
                signals = db.get_all_signals_for_csv()
                signal_buf = io.StringIO()
                w2 = csv.writer(signal_buf)
                if signals:
                    w2.writerow(list(signals[0].keys()))
                    for row in signals:
                        w2.writerow(list(row.values()))
                else:
                    w2.writerow(['signal_number', 'symbol', 'direction', 'setup_type',
                                 'entry_price', 'ml_score', 'lstm_score', 'xgboost_score',
                                 'sl_pips', 'rr_tp2', 'session', 'created_at'])
                    w2.writerow(['No setups yet.'])

                caption_lines = [
                    "All automated setups (%d records). Generated: %s UTC" % (
                        len(signals) if signals else 0, now_str)
                ]
                if hasattr(self, 'scheduler_obj') and self.scheduler_obj and hasattr(self.scheduler_obj, 'ml'):
                    ml_s = self.scheduler_obj.ml.get_model_status()
                    if ml_s.get('trained'):
                        caption_lines.append(
                            "Ensemble accuracy: %s | AUC: %s | Samples: %s" % (
                                ml_s.get('xgboost_accuracy', 'N/A'),
                                ml_s.get('xgboost_auc', 'N/A'),
                                ml_s.get('samples', 'N/A'),
                            )
                        )
                    else:
                        caption_lines.append("ML models: not yet trained (heuristic mode active).")

                await context.bot.send_document(
                    chat_id=telegram_id,
                    document=InputFile(
                        io.BytesIO(signal_buf.getvalue().encode('utf-8')),
                        filename="nixie_ml_setups_%s.csv" % now_str
                    ),
                    caption="\n".join(caption_lines),
                )
                self.logger.info(
                    "Admin download sent to user %d: %d trades, %d setups.",
                    telegram_id, len(trades) if trades else 0, len(signals) if signals else 0)
            else:
                self.logger.info(
                    "Download sent to user %d: %d trades (no admin access for setups).",
                    telegram_id, len(trades) if trades else 0)

            

        except Exception as e:
            self.logger.error("Download error for user %d: %s", telegram_id, e)
            await self._reply(
                update,
                f"Something went wrong. Please try again.\n\n"
                f"Contact {config.SUPPORT_CONTACT} if this continues.\n\n{config.FOOTER}"
            )


    # ==================== PUBLIC NOTIFICATION METHODS ====================

    async def cmd_test_briefing(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Admin only: Force-send the 06:30 daily briefing right now."""
        telegram_id = update.effective_user.id
        if telegram_id not in config.ADMIN_USER_IDS:
            await self._reply(update, "This command is for administrators only.")
            return
        await self._reply(update, "Sending daily briefing now...")
        if self.scheduler_obj:
            await self.scheduler_obj.trigger_daily_briefing_now(telegram_id)
        else:
            await self._reply(update, "Scheduler is not running.")

    async def cmd_test_news(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Admin only: Force-send the 08:00 news alert right now."""
        telegram_id = update.effective_user.id
        if telegram_id not in config.ADMIN_USER_IDS:
            await self._reply(update, "This command is for administrators only.")
            return
        await self._reply(update, "Sending news alert now...")
        if self.scheduler_obj:
            await self.scheduler_obj.trigger_news_alert_now(telegram_id)
        else:
            await self._reply(update, "Scheduler is not running.")

    async def cmd_test_weekly(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Admin only: Force-send the Sunday weekly analysis right now."""
        telegram_id = update.effective_user.id
        if telegram_id not in config.ADMIN_USER_IDS:
            await self._reply(update, "This command is for administrators only.")
            return
        await self._reply(update, "Sending weekly analysis now...")
        if self.scheduler_obj:
            await self.scheduler_obj.trigger_weekly_analysis_now(telegram_id)
        else:
            await self._reply(update, "Scheduler is not running.")

    async def cmd_test_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Admin only: Force-run a full market scan right now."""
        telegram_id = update.effective_user.id
        if telegram_id not in config.ADMIN_USER_IDS:
            await self._reply(update, "This command is for administrators only.")
            return
        await self._reply(update, "Running market scan now. This takes 1-3 minutes...")
        if self.scheduler_obj:
            await self.scheduler_obj.trigger_market_scan_now()
            await self._reply(update, "Market scan complete. Check your messages for any setups found.")
        else:
            await self._reply(update, "Scheduler is not running.")


    async def send_setup_alert(
        self,
        telegram_id: int,
        setup_data: dict,
        lot_size=None,
    ):
        """
        Send an automated setup alert to a specific user.
        Called by the scheduler when a new qualifying setup is detected.
        If the user is offline, the message is queued in the database.
        """
        try:
            message = utils.format_setup_message(
                signal_number=setup_data.get('signal_number', 0),
                symbol=setup_data['symbol'],
                direction=setup_data['direction'],
                setup_type=setup_data.get('setup_label', setup_data['setup_type']),
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
                lot_size=lot_size,
                expiry_hours=int(setup_data.get('expiry_hours', 8)),
            )
            sent = await self._send_with_retry(telegram_id, message)
            if not sent:
                db.queue_message(
                    telegram_id=telegram_id,
                    message_text=message,
                    message_type='SETUP_ALERT',
                )
        except Exception as e:
            self.logger.error(
                "Error sending setup alert to user %d: %s", telegram_id, e
            )

    async def send_trade_notification(self, telegram_id: int, message: str):
        """
        Send a trade or position update notification.
        If the user is offline, the message is queued.
        """
        try:
            clean = utils.validate_user_message(
                message + f"\n\n{config.FOOTER}"
            )
            sent = await self._send_with_retry(telegram_id, clean)
            if not sent:
                db.queue_message(
                    telegram_id=telegram_id,
                    message_text=clean,
                    message_type='TRADE_NOTIFICATION',
                )
        except Exception as e:
            self.logger.error(
                "Error sending trade notification to user %d: %s", telegram_id, e
            )


# ==================== UPGRADE / PAYMENT COMMANDS ====================

    async def cmd_upgrade(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /upgrade — show subscription plans with payment buttons."""
        if not self._check_rate_limit(update):
            await self._reply(update, "You are sending commands too quickly. Please wait a moment.")
            return

        telegram_id  = update.effective_user.id
        user         = db.get_user(telegram_id)
        sub_mgr      = get_subscription_manager()
        current_tier = sub_mgr.get_tier(telegram_id)

        from payment_handler import TIER_DISPLAY_NAMES
        tier_name = TIER_DISPLAY_NAMES.get(current_tier, current_tier.capitalize())

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "Basic - $30/month (1 account)",
                    callback_data='upgrade_basic_paystack',
                ),
            ],
            [
                InlineKeyboardButton(
                    "Pro - $100/month (3 accounts + any currency)",
                    callback_data='upgrade_pro_paystack',
                ),
            ],
            [
                InlineKeyboardButton(
                    "Pay via Stripe (card)",
                    callback_data='upgrade_basic_stripe',
                ),
                InlineKeyboardButton(
                    "Pay via Bybit (crypto)",
                    callback_data='upgrade_basic_bybit',
                ),
            ],
        ])

        await self._reply(
            update,
            utils.validate_user_message(
                "NIXIE TRADES SUBSCRIPTION PLANS\n\n"
                f"Your current plan: {tier_name}\n\n"
                "BASIC - $30 per month\n"
                "  - 1 MT5 account connection\n"
                "  - All automated setup alerts\n"
                "  - Auto-execution and position management\n\n"
                "PRO - $100 per month\n"
                "  - Up to 3 MT5 account connections\n"
                "  - All Basic features\n"
                "  - Accounts in any currency (USD, EUR, GBP, NGN, etc.)\n"
                "  - Priority support\n\n"
                "ADMIN - Free (staff only)\n"
                "  - Unlimited accounts\n"
                "  - All features\n\n"
                "Select a plan below to generate a secure payment link.\n"
                "Subscription is activated automatically within minutes of payment.\n\n"
                f"Questions? Contact {config.SUPPORT_CONTACT}\n\n"
                f"{config.FOOTER}"
            ),
            reply_markup=keyboard,
        )

    async def callback_upgrade_plan(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle upgrade plan + provider selection from inline keyboard."""
        query = update.callback_query
        await query.answer()

        telegram_id = query.from_user.id
        parts       = query.data.split('_')
        if len(parts) < 3:
            await query.edit_message_text("Invalid selection. Please use /upgrade again.")
            return

        tier     = parts[1]   # 'basic' or 'pro'
        provider = parts[2]   # 'paystack', 'stripe', 'bybit'

        sub_mgr  = get_subscription_manager()
        info     = sub_mgr.generate_payment_link(
            telegram_id=telegram_id,
            tier=tier,
            provider=provider,
        )

        if info and info.get('url'):
            from payment_handler import TIER_PRICES_USD, TIER_DISPLAY_NAMES
            amount    = TIER_PRICES_USD.get(tier, 0)
            plan_name = TIER_DISPLAY_NAMES.get(tier, tier.capitalize())
            provider_label = {'paystack': 'Paystack', 'stripe': 'Stripe', 'bybit': 'Bybit'}.get(provider, provider)

            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton(
                    f"Pay ${amount}/month via {provider_label}",
                    url=info['url'],
                ),
            ]])
            await query.edit_message_text(
                utils.validate_user_message(
                    "PAYMENT LINK READY\n\n"
                    f"Plan:      {plan_name}\n"
                    f"Amount:    ${amount} per month\n"
                    f"Provider:  {provider_label}\n\n"
                    "Tap the button below to complete your payment securely.\n\n"
                    "Your subscription activates automatically within a few minutes "
                    "of payment confirmation. You will receive a Telegram notification.\n\n"
                    f"Reference: {info.get('reference', 'N/A')}\n\n"
                    f"Support: {config.SUPPORT_CONTACT}\n\n"
                    f"{config.FOOTER}"
                ),
                reply_markup=keyboard,
            )
        else:
            await query.edit_message_text(
                utils.validate_user_message(
                    "Payment link could not be generated at this time.\n\n"
                    f"Please contact {config.SUPPORT_CONTACT} for assistance.\n\n"
                    f"{config.FOOTER}"
                )
            )


# ==================== ENTRY POINT ====================

if __name__ == '__main__':
    bot = NixTradesBot()
    bot.run()
