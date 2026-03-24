import asyncio
import io
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

try:
    import httpx as _httpx_check
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "httpx not installed. LLM weekly analysis will be skipped. "
        "Run: pip install httpx --break-system-packages"
    )

import pandas as pd
from telegram import InputFile
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from telegram import Bot
from telegram.error import Forbidden, NetworkError

import config
import database as db
import utils
from ml_models import MLEnsemble
from mt5_connector import MT5Connector
from news_fetcher import NewsFetcher
from smc_strategy import SMCStrategy

logger = logging.getLogger(__name__)

# Telegram allows 30 messages per second globally.
# At 0.04s per message we stay safely under the limit.
_TELEGRAM_SEND_DELAY = 0.04   # seconds between each bot.send_message call


class NixTradesScheduler:
    """
    Runs recurring background tasks:
      - Daily 8 AM UTC market overview with news summary
      - Market scan on configured interval for new SMC setups
      - Per-user timed alerts in each user's local timezone
      - Periodic news cache refresh
    """

    def __init__(
        self,
        telegram_bot:  Bot,
        mt5_connector: MT5Connector,
        smc_strategy:  SMCStrategy,
        ml_ensemble:   MLEnsemble,
        news_fetcher:  NewsFetcher,
        position_monitor=None,
    ):
        self.logger   = logging.getLogger(f"{__name__}.NixTradesScheduler")
        self.bot      = telegram_bot
        self.mt5      = mt5_connector
        self.smc      = smc_strategy
        self.ml       = ml_ensemble
        self.news     = news_fetcher
        self.monitor  = position_monitor   # Set after position monitor starts

        self.scheduler = AsyncIOScheduler()
        self.running   = False

        # Stores the most recent weekly LLM bias per symbol.
        # Updated by _send_weekly_analysis every Sunday.
        # Read by _scan_symbol as an advisory confirmation layer.
        # Format: { 'EURUSD': 'BUY', 'GBPUSD': 'SELL', 'XAUUSD': 'NEUTRAL', ... }
        self._llm_weekly_bias: dict = {}

        # Stores most recent daily briefing context string (plain text).
        # Appended to the LLM prompt in weekly analysis for continuity.
        self._last_daily_context: str = ''

        # Tracks which users already received each message today.
        # Format: "YYYY-MM-DD" -> set of telegram_ids
        self._briefing_sent: dict = {}
        self._news_sent:     dict = {}
        self._weekly_sent:   dict = {}

        self.market_scan_interval_minutes = max(
            1, int(getattr(config, 'MARKET_SCAN_INTERVAL_MINUTES', 15))
        )
        self.alert_check_interval_minutes = max(
            1, int(getattr(config, 'ALERT_CHECK_INTERVAL_MINUTES', 5))
        )
        self.news_update_interval_minutes = max(
            5, int(getattr(config, 'NEWS_UPDATE_INTERVAL_MINUTES', 120))
        )

        # Chart generator for setup alert images
        try:
            from chart_generator import ChartGenerator
            self._chart_gen: object = ChartGenerator()
            self.logger.info("ChartGenerator loaded.")
        except Exception as _cg_err:
            self._chart_gen = None
            self.logger.warning(
                "ChartGenerator unavailable: %s. "
                "Setup alerts will be sent as text only.", _cg_err)

        # Tracks which high-impact news events have already had a 30-min reminder sent.
        # Key: (currency, title_lower[:50], rounded_timestamp_isoformat)
        self._reminded_events: set = set()

        self.logger.info("Scheduler initialised.")

    @staticmethod
    def _candles_to_df(raw: List[dict]) -> pd.DataFrame:
        """
        Convert MT5 candle payload to a UTC-indexed DataFrame.
        MT5 worker returns Unix timestamps in seconds.
        """
        df = pd.DataFrame(raw)
        if 'time' not in df.columns:
            raise ValueError("Candle payload missing 'time' field.")

        ts_num = pd.to_numeric(df['time'], errors='coerce')
        if ts_num.notna().sum() >= max(1, int(len(df) * 0.8)):
            df['time'] = pd.to_datetime(ts_num, unit='s', utc=True, errors='coerce')
        else:
            df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')

        df.dropna(subset=['time'], inplace=True)
        if df.empty:
            raise ValueError("No valid candle timestamps after parsing.")

        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        return df

    # ==================== LIFECYCLE ====================

    def start(self):
        """Start the APScheduler. Must be called from inside the running event loop."""
        try:
            
            self.scheduler.add_job(
                self.scan_markets,
                IntervalTrigger(
                    minutes=self.market_scan_interval_minutes,
                    timezone='UTC',
                ),
                id='market_scan',
                name='Market Scan',
                replace_existing=True,
                misfire_grace_time=max(60, self.market_scan_interval_minutes * 120),
                coalesce=True,
                max_instances=1,
            )

            # Per-user timezone dispatcher — checks every 5 minutes
            # and sends the 06:30 briefing, 08:00 news, Sunday 09:00 weekly
            # to each user at the right time in their own timezone.
            self.scheduler.add_job(
                self._timed_messages,
                IntervalTrigger(
                    minutes=self.alert_check_interval_minutes,
                    timezone='UTC',
                ),
                id='timed_messages',
                name='Per-User Timed Message Dispatcher',
                replace_existing=True,
                misfire_grace_time=max(60, self.alert_check_interval_minutes * 120),
                coalesce=True,
                max_instances=1,
            )

            self.scheduler.add_job(
                self._refresh_news_cache,
                IntervalTrigger(
                    minutes=self.news_update_interval_minutes,
                    timezone='UTC',
                ),
                id='news_refresh',
                name='News Cache Refresh',
                replace_existing=True,
                misfire_grace_time=max(120, self.news_update_interval_minutes * 120),
                coalesce=True,
                max_instances=1,
            )

            self.scheduler.add_job(
                self._run_reconciliation,
                IntervalTrigger(minutes=30, timezone='UTC'),
                id='trade_reconciliation',
                name='Open Trade Reconciliation',
                replace_existing=True,
                misfire_grace_time=300,
                coalesce=True,
                max_instances=1,
            )

            self.scheduler.add_job(
                self._check_news_reminders,
                IntervalTrigger(minutes=5, timezone='UTC'),
                id='news_reminders',
                name='30-Minute News Reminders',
                replace_existing=True,
                misfire_grace_time=120,
                coalesce=True,
                max_instances=1,
            )

            self.scheduler.start()
            self.running = True
            self.logger.info(
                "Scheduler started.\n"
                "  Per-user alerts (6:30 AM briefing, 8:00 AM news, Sunday analysis): checked every %d minutes.\n"
                "  Market scan: every %d minutes.\n"
                "  News cache refresh: every %d minutes.",
                self.alert_check_interval_minutes,
                self.market_scan_interval_minutes,
                self.news_update_interval_minutes,
            )
        except Exception as e:
            self.logger.error("Failed to start scheduler: %s", e)

    def stop(self):
        """Stop the scheduler. wait=False avoids blocking on running jobs."""
        try:
            if self.scheduler.running:
                self.scheduler.shutdown(wait=False)
            self.running = False
            self.logger.info("Scheduler stopped.")
        except Exception as e:
            self.logger.error("Error stopping scheduler: %s", e)
            
    # ==================== TRADE RECONCILIATION ====================

    async def _reconcile_open_trades(self):
        """
        Cross-check every OPEN database trade against MT5.

        Three outcomes per trade:
          - Ticket still open in MT5:  leave it alone.
          - Ticket closed in MT5:      mark CLOSED with outcome + pips.
          - Ticket not found in MT5:   mark EXPIRED (pending order lapsed).

        Runs every 30 minutes so the database never stays out of sync for long.
        This handles the case where:
          a) The pending order expiry window passed and MT5 cancelled it.
          b) The bot was restarted mid-trade and missed the close event.
        """
        try:
            open_trades = db.get_open_trades_all_users()
            if not open_trades:
                self.logger.debug("Reconciliation: no open trades to check.")
                return

            self.logger.info(
                "Reconciliation: checking %d open trade(s) against MT5.",
                len(open_trades),
            )

            for trade in open_trades:
                ticket      = trade.get('mt5_ticket')
                trade_id    = trade.get('id')
                telegram_id = trade.get('telegram_id')

                if not ticket or not telegram_id:
                    continue

                try:
                    status_info = await self.mt5.check_ticket_status(telegram_id, ticket)
                    status      = status_info.get('status', 'UNKNOWN')

                    if status == 'POSITION':
                        # Still open — nothing to do
                        pass

                    elif status == 'PENDING':
                        # Pending order still waiting — nothing to do
                        pass

                    elif status == 'CLOSED':
                        profit_pips  = float(status_info.get('profit_pips',  0) or 0)
                        close_price  = float(status_info.get('close_price',  0) or 0)
                        closed_at    = status_info.get('closed_at', '')
                        realized_pnl = status_info.get('realized_pnl')
                        realized_pnl = float(realized_pnl) if realized_pnl is not None else None
                        # Use realized_pnl sign for outcome when available because
                        # profit_pips magnitude can be approximate for cross pairs.
                        if realized_pnl is not None:
                            outcome = (
                                'WIN'       if realized_pnl > 0 else
                                ('LOSS'     if realized_pnl < 0 else
                                 'BREAKEVEN')
                            )
                        else:
                            outcome = (
                                'WIN'       if profit_pips > 0 else
                                ('LOSS'     if profit_pips < 0 else
                                 'BREAKEVEN')
                            )
                        db.mark_trade_reconciled(
                            ticket, outcome, profit_pips,
                            close_price, closed_at,
                            realized_pnl=realized_pnl,
                        )
                        self.logger.info(
                            "Reconciliation: ticket %d CLOSED offline — "
                            "outcome=%s pips=%.1f pnl=%s.",
                            ticket, outcome, profit_pips,
                            ('%.2f' % realized_pnl) if realized_pnl is not None else 'N/A',
                        )

                    elif status in ('NOT_FOUND', 'UNKNOWN'):
                        db.mark_trade_expired(trade_id)
                        self.logger.info(
                            "Reconciliation: ticket %d not found in MT5 — "
                            "marked EXPIRED.", ticket,
                        )

                except Exception as trade_err:
                    self.logger.warning(
                        "Reconciliation: error checking ticket %d: %s",
                        ticket, trade_err,
                    )

        except Exception as e:
            self.logger.error("Reconciliation job failed: %s", e, exc_info=True)

    async def _run_reconciliation(self):
        """APScheduler entry point for the reconciliation job."""
        if not await self.mt5.is_worker_reachable():
            self.logger.debug(
                "Reconciliation skipped: MetaApi not reachable.")
            return
        await self._reconcile_open_trades()

    # ==================== DAILY 8 AM ALERT ====================

    async def daily_alert(self):
        """
        Daily market overview sent at 8:00 AM UTC to all subscribed users.
        Contains market structure summary and high-impact news for the day.
        """
        try:
            self.logger.info("Running daily 8 AM alert.")
            subscribed = db.get_subscribed_users()
            if not subscribed:
                self.logger.info("No subscribed users for daily alert.")
                return

            market_overview = await self._generate_market_overview()
            news_summary    = self.news.format_news_summary(hours_ahead=24)

            message = utils.validate_user_message(
                f"GOOD MORNING - DAILY MARKET OVERVIEW\n"
                f"{datetime.now(timezone.utc).strftime('%A, %B %d, %Y')} (UTC)\n\n"
                f"{market_overview}\n\n"
                f"HIGH-IMPACT NEWS TODAY:\n"
                f"{news_summary}\n\n"
                f"The system scans all pairs every {self.market_scan_interval_minutes} minutes. "
                f"Qualifying setups will be sent automatically when detected.\n\n"
                f"{config.FOOTER}"
            )

            sent = 0
            for user in subscribed:
                try:
                    await self.bot.send_message(
                        chat_id=user['telegram_id'], text=message)
                    sent += 1
                except Forbidden:
                    self.logger.info(
                        "User %d has blocked the bot. Skipping.",
                        user['telegram_id'])
                except NetworkError as e:
                    self.logger.warning(
                        "Network error sending daily alert to user %d: %s",
                        user['telegram_id'], e)
                except Exception as e:
                    self.logger.error(
                        "Error sending daily alert to user %d: %s",
                        user['telegram_id'], e)
                finally:
                    await asyncio.sleep(_TELEGRAM_SEND_DELAY)

            self.logger.info(
                "Daily alert sent to %d of %d subscribed users.", sent, len(subscribed))
        except Exception as e:
            self.logger.error("Fatal error in daily_alert: %s", e, exc_info=True)

    async def _refresh_news_cache(self):
        """Periodically prefetch news windows used by scans and alerts."""
        try:
            loop = asyncio.get_running_loop()
            windows = (2, 24, 168)
            counts = {}
            for hours in windows:
                events = await loop.run_in_executor(
                    None, self.news.get_red_folder_events, hours
                )
                counts[hours] = len(events) if events else 0
            self.logger.info(
                "News cache refreshed: 2h=%d 24h=%d 168h=%d",
                counts.get(2, 0),
                counts.get(24, 0),
                counts.get(168, 0),
            )
        except Exception as e:
            self.logger.error("News cache refresh failed: %s", e)

    # ==================== 30-MINUTE NEWS REMINDERS ====================

    async def _check_news_reminders(self):
        """
        Checks every 5 minutes for HIGH-impact news events that are
        25-35 minutes away and sends a single advance warning to all
        subscribed users. Each event is reminded about only once per
        occurrence (tracked via self._reminded_events).
        """
        try:
            loop   = asyncio.get_running_loop()
            events = await loop.run_in_executor(
                None, self.news.get_red_folder_events, 1
            )
            if not events:
                return

            now_utc = datetime.now(timezone.utc)

            for event in events:
                ev_ts = event.timestamp
                if ev_ts.tzinfo is None:
                    ev_ts = ev_ts.replace(tzinfo=timezone.utc)
                else:
                    ev_ts = ev_ts.astimezone(timezone.utc)

                minutes_until = (ev_ts - now_utc).total_seconds() / 60.0

                # Only remind when 25-35 minutes remain
                if not (25.0 <= minutes_until <= 35.0):
                    continue

                # Deduplication key: currency + title + rounded minute
                event_key = (
                    event.currency.upper().strip(),
                    event.title.strip().lower()[:60],
                    ev_ts.replace(second=0, microsecond=0).isoformat(),
                )
                if event_key in self._reminded_events:
                    continue

                self._reminded_events.add(event_key)

                # Cap set size to prevent unbounded growth across long uptimes
                if len(self._reminded_events) > 300:
                    self._reminded_events = set(list(self._reminded_events)[-200:])

                time_str = utils.calculate_time_until(ev_ts)
                reminder = utils.validate_user_message(
                    "HIGH-IMPACT NEWS REMINDER\n\n"
                    f"Event:    {event.currency} — {event.title}\n"
                    f"Time:     In approximately {time_str}\n\n"
                    f"Avoid opening new trades on pairs involving {event.currency} "
                    "for the next 30 minutes.\n\n"
                    "The bot pauses automated setups on affected pairs "
                    "during the blackout window.\n\n"
                    f"{config.FOOTER}"
                )

                from payment_handler import get_subscription_manager as _gsm
                _sub_mgr   = _gsm()
                subscribed = db.get_subscribed_users()
                for user in subscribed:
                    if _sub_mgr.get_tier(user['telegram_id']) in ('basic', 'pro', 'admin'):
                        await self._safe_send(user['telegram_id'], reminder)
                        await asyncio.sleep(_TELEGRAM_SEND_DELAY)

                self.logger.info(
                    "News reminder sent: %s %s in %.0f minutes.",
                    event.currency, event.title, minutes_until,
                )

        except Exception as e:
            self.logger.error("Error in _check_news_reminders: %s", e, exc_info=True)

    # ==================== MARKET SCAN ====================

    async def scan_markets(self):
        """Scan all monitored symbols on the configured scheduler interval."""
        try:
            self.logger.info("Starting market scan.")
            if not await self.mt5.is_worker_reachable():
                self.logger.warning(
                    "MetaApi not reachable. Skipping market scan. "
                    "Check Windows VPS worker is running.")
                return

            for symbol in config.MONITORED_SYMBOLS:
                try:
                    await self._scan_symbol(symbol)
                    await asyncio.sleep(0.5)   # Brief pause between symbols
                except Exception as e:
                    self.logger.error("Error scanning %s: %s", symbol, e)

            self.logger.info("Market scan complete.")
        except Exception as e:
            self.logger.error("Fatal error in scan_markets: %s", e, exc_info=True)

    async def _scan_symbol(self, symbol: str):
        """
        Full 4-phase analysis for one symbol.
        Phase 1: D1 trend context
        Phase 2: H1 structure (BOS / MSS)
        Phase 3: ML validation
        Phase 4: Entry/SL/TP calculation and broadcast
        """
        try:
            # Bail immediately if the MT5 worker is offline.
            # Without this, the scan blocks for 10+ minutes per symbol
            # retrying 4 times per timeframe with no worker available.
            if not await self.mt5.is_worker_reachable():
                self.logger.info(
                    "MetaApi not reachable. Skipping %s.", symbol)
                return

            # Check news blackout window before doing any analysis
            is_blackout, news_event = self.news.check_news_blackout(symbol)
            if is_blackout:
                self.logger.info(
                    "Skipping %s: news blackout for %s %s.",
                    symbol,
                    getattr(news_event, 'currency', ''),
                    getattr(news_event, 'title', ''))
                return

            # 15-minute guard: prevents reprocessing the exact same pair
            # on back-to-back scan cycles before any new structure can form.
            # The 480-minute per-direction guard is applied later after the
            # trade direction is determined from the POI.
            if db.recent_signal_exists(symbol, minutes=15):
                self.logger.debug(
                    "Skipping %s: signal sent within the last 15 minutes.", symbol)
                return

            # Fetch multi-timeframe data
            # D1 provides HTF context, H4 provides intermediate confirmation,
            # H1 provides structure, M15 provides entry timing, M5 provides
            # precise entry refinement within the H1 POI zone.
            # Candle counts are sized to the minimum each operation actually consumes:
            #   D1:  _identify_swings uses tail(100), lookback=2 needs 104 minimum.
            #         300 gives robust swing history across different market phases.
            #   H4:  Same as D1. 400 covers ~67 trading days for alignment context.
            #   H1:  OB rolling_avg_vol needs 20 bars, OB loop needs 10+6=16 bars,
            #         detect_break_of_structure uses tail(50). 600 = ~25 trading days.
            #   M15: Inducement lookback 80 bars (20 hours). ML features use 100 bars.
            #         700 bars = ~175 hours of M15 data for POI and sweep detection.
            #   M5:  OB refinement uses tail(80). 400 bars = 33 hours of M5 data.
            d1_raw  = await self.mt5.get_historical_data(symbol, 'D1',  bars=300)
            h4_raw  = await self.mt5.get_historical_data(symbol, 'H4',  bars=400)
            h1_raw  = await self.mt5.get_historical_data(symbol, 'H1',  bars=600)
            m15_raw = await self.mt5.get_historical_data(symbol, 'M15', bars=700)
            m5_raw  = await self.mt5.get_historical_data(symbol, 'M5',  bars=400)

            # Minimum bar checks are set to what each algorithm actually requires.
            # D1: _identify_swings uses tail(100) — need at least 105 bars.
            if not d1_raw or len(d1_raw) < 105:
                self.logger.debug(
                    "Insufficient D1 data for %s (%d bars). Need 105. Skipping.",
                    symbol, len(d1_raw) if d1_raw else 0)
                return
            # H1: detect_break_of_structure uses tail(50), OB detection needs 36+.
            #     200 bars is a conservative safe minimum for all H1 operations.
            if not h1_raw or len(h1_raw) < 200:
                self.logger.debug(
                    "Insufficient H1 data for %s (%d bars). Need 200. Skipping.",
                    symbol, len(h1_raw) if h1_raw else 0)
                return
            # M15: inducement lookback is 80 bars, ML features need 100 bars.
            if not m15_raw or len(m15_raw) < 120:
                self.logger.debug(
                    "Insufficient M15 data for %s (%d bars). Need 120. Skipping.",
                    symbol, len(m15_raw) if m15_raw else 0)
                return
            # M5: OB refinement uses tail(80). 400 bars = 33 hours of M5 data.
            if not m5_raw or len(m5_raw) < 400:
                self.logger.debug(
                    "Insufficient M5 data for %s (%d bars). Need 400. Skipping.",
                    symbol, len(m5_raw) if m5_raw else 0)
                return

            d1_df  = self._candles_to_df(d1_raw)
            h4_df  = self._candles_to_df(h4_raw) if h4_raw else None
            h1_df  = self._candles_to_df(h1_raw)
            m15_df = self._candles_to_df(m15_raw)
            m5_df  = self._candles_to_df(m5_raw) if m5_raw else None

            # Phase 1: HTF trend context (D1)
            htf_trend = self.smc.determine_htf_trend(d1_df)
            if htf_trend.get('trend') == 'RANGING':
                self.logger.info("%s D1 trend is RANGING. Skipping.", symbol)
                return

            # H4 confirmation is stored here but the gate is NOT applied yet.
            # setup_type is not known until after BOS/MSS detection below.
            # The gate only applies to BOS (continuation) setups.
            # MSS (reversal) setups are inherently counter-trend and must
            # not be blocked by H4 alignment.
            h4_aligned    = False
            h4_trend_data = None
            if h4_df is not None and len(h4_df) >= 50:
                h4_trend_data = self.smc.determine_htf_trend(h4_df)
                h4_aligned    = h4_trend_data.get('trend') == htf_trend.get('trend')
                
            # Phase 2: Intermediate structure (H1)
            bos_events = self.smc.detect_break_of_structure(h1_df, htf_trend['trend'])
            mss_event  = self.smc.detect_market_structure_shift(
                h1_df, htf_trend['trend'], symbol=symbol)

            setup_type      = None
            poi             = None
            all_candidates: list = []

            if len(bos_events) >= 2:
                # BOS continuation: Priority 1 = Breaker Block, Fallback = Order Block.
                # Per PRD 3.2: in a strong double-BOS trend, BBs are more reliable.
                # OBs are only used when no valid BBs exist.
                setup_type = 'BOS'
                breakers   = self.smc.detect_breaker_blocks(
                    h1_df, htf_trend['trend'],
                    htf_trend.get('swing_high', 0),
                    htf_trend.get('swing_low', 0),
                    symbol=symbol)
                if breakers:
                    all_candidates = breakers
                    self.logger.debug(
                        "%s BOS: %d Breaker Block(s) found. "
                        "Using BB as primary candidates.",
                        symbol, len(breakers))
                else:
                    obs = self.smc.detect_order_blocks(
                        h1_df, htf_trend['trend'], symbol=symbol)
                    all_candidates = obs
                    self.logger.info(
                        "%s BOS: No Breaker Blocks found. "
                        "Falling back to %d Order Block(s).",
                        symbol, len(obs))

            elif mss_event:
                # MSS reversal: Priority 1 = Order Block, Fallback = Breaker Block.
                # Per PRD 3.1: algorithm prioritizes the extreme OB candle responsible
                # for the MSS displacement. BB is only used if no valid OB exists.
                setup_type = 'MSS'
                obs = self.smc.detect_order_blocks(
                    h1_df, mss_event['direction'], symbol=symbol)
                if obs:
                    all_candidates = obs
                    self.logger.debug(
                        "%s MSS: %d Order Block(s) found. "
                        "Using OB as primary candidates.",
                        symbol, len(obs))
                else:
                    breakers = self.smc.detect_breaker_blocks(
                        h1_df, mss_event['direction'],
                        htf_trend.get('swing_high', 0),
                        htf_trend.get('swing_low', 0),
                        symbol=symbol)
                    all_candidates = breakers
                    self.logger.info(
                        "%s MSS: No Order Blocks found. "
                        "Falling back to %d Breaker Block(s).",
                        symbol, len(breakers))

            if not all_candidates or setup_type is None:
                self.logger.info(
                    "%s: no valid structure or POI candidates found.", symbol)
                return

            # Apply the H4 alignment gate now that setup_type is known.
            # BOS is a trend continuation trade and requires H4 to agree.
            # MSS is a reversal trade; requiring H4 alignment on a reversal
            # is a logical contradiction and must be skipped.
            if setup_type == 'BOS' and h4_trend_data is not None:
                if not h4_aligned and h4_trend_data.get('trend') != 'RANGING':
                    self.logger.info(
                        "%s BOS: H4 trend (%s) opposes D1 trend (%s). "
                        "BOS setups require HTF alignment. Skipping.",
                        symbol,
                        h4_trend_data.get('trend'),
                        htf_trend.get('trend'),
                    )
                    return
            elif setup_type == 'MSS':
                self.logger.debug(
                    "%s MSS: H4 alignment check skipped. "
                    "MSS is a reversal trade.", symbol)

            # Filter out any POI that has already been mitigated by price action.
            # A mitigated zone means institutions already filled their orders there;
            # there is no point targeting an empty well.
            # Breaker Blocks use touch_mitigation=True: a wick into the zone
            # followed by a rejection does NOT invalidate it.
            # Order Blocks use the default strict boundary rule.
            unmitigated = []
            for _cand in all_candidates:
                _is_bb = str(_cand.get('type', 'OB')).upper() in ('BB', 'BREAKER')
                if not self.smc.is_poi_mitigated(
                    _cand, h1_df,
                    touch_mitigation=_is_bb,
                    symbol=symbol,
                ):
                    unmitigated.append(_cand)

            if not unmitigated:
                self.logger.info(
                    "%s: all %d detected POI(s) are mitigated. "
                    "No valid entry zone remaining. Skipping.",
                    symbol, len(all_candidates))
                return

            # Use the highest-confidence unmitigated POI as the anchor for
            # inducement detection. We need one reference zone to define the
            # direction and price area of the expected sweep.
            anchor_poi      = max(unmitigated, key=lambda x: x.get('confidence', 0))
            trade_direction = anchor_poi['direction']

            # Phase 2b: Inducement gate — validated on H1 to match setup timeframe.
            # H1 identifies the trade idea (BOS/MSS and POI zones).
            # M15 confirms the liquidity sweep (inducement) has actually occurred.
            # M5 is used later for precise entry price refinement only.
            # This 3-layer zoom is the correct Smart Money workflow.
            # Pass the full M15 DataFrame so the inducement detector can scan
            # 80 bars (20 hours) of M15 history for the liquidity sweep.
            # The previous tail(60) combined with the internal tail(40) was
            # producing only 40 bars (10 hours), missing sweeps on slow setups.
            # An H1 BOS can precede the sweep by up to 12-16 hours.
            inducement = self.smc.detect_inducement_post_structure(
                m15_df,
                anchor_poi,
                trade_direction,
                lookback_bars=80,
                symbol=symbol,
                timeframe='M15',
            )
            if inducement is None:
                self.logger.info(
                    "AWAITING INDUCEMENT | %s | %s %s | "
                    "Anchor POI [%.5f - %.5f] valid but no M15 sweep yet. "
                    "Will re-check next scan.",
                    symbol, setup_type, trade_direction,
                    float(anchor_poi.get('low', 0)), float(anchor_poi.get('high', 0)),
                )
                return

            sweep_level = float(inducement.get('sweep_level', 0))

            # Direction is now confirmed from the anchor POI.
            # Apply the 480-minute per-direction cooldown here so that a SELL
            # can fire on a pair that already had a BUY in the last 8 hours.
            _dir_str = 'BUY' if trade_direction == 'BULLISH' else 'SELL'
            if db.recent_signal_exists(symbol, minutes=480, direction=_dir_str):
                self.logger.info(
                    "COOLDOWN | %s %s | same direction sent within 480 minutes. "
                    "Skipping.", symbol, _dir_str)
                return

            # Now that the sweep is confirmed, pick the CLOSEST unmitigated POI
            # to the sweep level. This is the real entry zone — the nearest
            # institutional block to where the stop hunt just occurred.
            poi = self.smc.get_closest_unmitigated_poi(
                unmitigated, sweep_level, trade_direction, h1_df
            )
            if poi is None:
                # Fallback: no POI is optimally positioned relative to the sweep,
                # so use the anchor that we already validated above.
                poi = anchor_poi
                self.logger.info(
                    "%s: No POI closer to sweep %.5f found. "
                    "Falling back to anchor POI [%.5f - %.5f].",
                    symbol, sweep_level,
                    float(anchor_poi.get('low', 0)), float(anchor_poi.get('high', 0)),
                )
            else:
                poi_mid = (float(poi.get('low', 0)) + float(poi.get('high', 0))) / 2
                self.logger.info(
                    "INDUCEMENT CONFIRMED | %s | %.1f pip sweep | Quality: %s | "
                    "Entry POI [%.5f - %.5f] (%.5f from sweep).",
                    symbol,
                    inducement.get('sweep_pips', 0),
                    inducement.get('quality', 'N/A'),
                    float(poi.get('low', 0)), float(poi.get('high', 0)),
                    abs(poi_mid - sweep_level),
                )

            # Phase 3: ML validation
            ml_result     = self.ml.get_ensemble_prediction(
                m15_df, poi, htf_trend, setup_type)
            consensus     = ml_result['consensus_score']
            should_send, tier = self.ml.should_send_setup(consensus)

            if not should_send:
                self.logger.info(
                    "%s: ML consensus %d%% below threshold. Rejected.", symbol, consensus)
                return

            # ATR and session filters
            if not self._check_filters(symbol, m15_df, htf_trend, poi, tier):
                return

            # Phase 4: Entry / SL / TP calculation
            entry_cfg = self.smc.calculate_entry_price(
                poi, 'UNICORN' if tier == 'PREMIUM' else 'STANDARD', consensus)
            sl_cfg    = self.smc.calculate_stop_loss(
                poi, poi['direction'], symbol,
                self.smc._calculate_atr(m15_df.tail(20)))
            
            # Multi-timeframe sniper entry refinement.
            # Correct SMC sequence: drill down from H1 POI -> M15 OB -> M5 OB.
            #
            # ENTRY BOUNDARY RULE (Elite SMC):
            #   BUY demand zone: price pulls back DOWN into zone.
            #     Price enters through the TOP (poi_high).
            #     Limit order at m_ob['high'] — fills on the first pip that
            #     touches the lower-TF demand zone.
            #   SELL supply zone: price rallies UP into zone.
            #     Price enters through the BOTTOM (poi_low).
            #     Limit order at m_ob['low'] — fills on first touch of supply.
            #
            # CRITICAL: When entry is refined to a lower-TF OB, the SL MUST also
            # be recalculated from that same lower-TF OB. The H1 POI SL is too
            # wide relative to the tighter M15/M5 entry boundary. Keeping the
            # H1 SL was producing poor RR that the TP filter then rejected,
            # suppressing valid sniper setups.
            _poi_high   = float(poi.get('high', entry_cfg['entry_price']))
            _poi_low    = float(poi.get('low',  entry_cfg['entry_price']))
            _is_buy     = poi['direction'] == 'BULLISH'
            _refined    = False
            _refined_ob = None   # The lower-TF OB selected for entry

            # Priority 1: M15 OB within the H1 POI zone.
            if m15_df is not None and len(m15_df) >= 50:
                try:
                    m15_obs = self.smc.detect_order_blocks(
                        m15_df.tail(80), poi['direction'], symbol=symbol)
                    for m15_ob in m15_obs[:5]:
                        boundary = (
                            float(m15_ob.get('high', 0))
                            if _is_buy
                            else float(m15_ob.get('low', 0))
                        )
                        if _poi_low <= boundary <= _poi_high and boundary > 0:
                            self.logger.info(
                                "M15 OB sniper entry for %s: zone %.5f -> M15 boundary %.5f",
                                symbol, entry_cfg['entry_price'], round(boundary, 5))
                            entry_cfg['entry_price'] = round(boundary, 5)
                            _refined_ob = m15_ob
                            _refined    = True
                            break
                except Exception as m15_err:
                    self.logger.debug(
                        "M15 refinement skipped for %s: %s", symbol, m15_err)

            # Priority 2: M5 OB within H1 POI zone (only if no M15 OB found).
            if not _refined and m5_df is not None and len(m5_df) >= 50:
                try:
                    m5_obs = self.smc.detect_order_blocks(
                        m5_df.tail(80), poi['direction'], symbol=symbol)
                    for m5_ob in m5_obs[:5]:
                        boundary = (
                            float(m5_ob.get('high', 0))
                            if _is_buy
                            else float(m5_ob.get('low', 0))
                        )
                        if _poi_low <= boundary <= _poi_high and boundary > 0:
                            self.logger.info(
                                "M5 OB sniper entry for %s: zone %.5f -> M5 boundary %.5f",
                                symbol, entry_cfg['entry_price'], round(boundary, 5))
                            entry_cfg['entry_price'] = round(boundary, 5)
                            _refined_ob = m5_ob
                            _refined    = True
                            break
                except Exception as m5_err:
                    self.logger.debug(
                        "M5 refinement skipped for %s: %s", symbol, m5_err)

            # Recalculate SL from the lower-TF OB that was selected for entry.
            # Without this, an M15 entry at 1.0852 could have an SL at the H1
            # POI low of 1.0810 — a 42-pip stop on a 5-pip zone entry.
            # The SL should sit just beyond the OPPOSITE end of the refined OB:
            #   BUY:  SL below the M15/M5 OB low  (+ ATR structural buffer)
            #   SELL: SL above the M15/M5 OB high (+ ATR structural buffer)
            if _refined and _refined_ob is not None:
                _refinement_atr = self.smc._calculate_atr(m15_df.tail(20))
                sl_cfg = self.smc.calculate_stop_loss(
                    _refined_ob,
                    poi['direction'],
                    symbol,
                    _refinement_atr,
                )
                self.logger.info(
                    "SL recalculated from lower-TF OB for %s: "
                    "H1-POI SL %.5f -> refined OB SL %.5f (%.1f pips tighter).",
                    symbol,
                    float(poi.get('low', 0)) if _is_buy else float(poi.get('high', 0)),
                    sl_cfg['stop_loss'],
                    abs(
                        (float(poi.get('low', 0)) if _is_buy else float(poi.get('high', 0)))
                        - sl_cfg['stop_loss']
                    ) / utils.get_pip_value(symbol),
                )
            else:
                self.logger.debug(
                    "No lower-TF OB found within H1 POI [%.5f - %.5f] for %s. "
                    "Using H1 zone-percentage entry %.5f with H1 POI SL %.5f.",
                    _poi_low, _poi_high, symbol,
                    entry_cfg['entry_price'], sl_cfg['stop_loss'])
            try:
                tp_cfg = self.smc.calculate_take_profits(
                    entry_cfg['entry_price'],
                    sl_cfg['stop_loss'],
                    poi['direction'],
                    htf_trend.get(
                        'swing_high' if poi['direction'] == 'BULLISH' else 'swing_low',
                        0
                    ),
                    symbol,
                )
            except ValueError as rr_err:
                self.logger.info(
                    "Setup rejected on RR filter for %s: %s", symbol, rr_err)
                return

            # News warning text
            next_news = self.news.get_next_high_impact(symbol)
            news_warn = ''
            if next_news:
                event_ts = next_news.timestamp
                if event_ts.tzinfo is None:
                    event_ts = event_ts.replace(tzinfo=timezone.utc)
                else:
                    event_ts = event_ts.astimezone(timezone.utc)
                mins = int((event_ts - datetime.now(timezone.utc)).total_seconds() / 60)
                if 0 < mins <= 480:
                    _time_str = utils.calculate_time_until(event_ts)
                    news_warn = (
                        "Note: %s %s in approximately %s. "
                        "Consider reducing position size or waiting for the release." % (
                            next_news.currency,
                            next_news.title,
                            _time_str,
                        )
                    )

            setup_tier  = 'UNICORN' if tier == 'PREMIUM' else 'STANDARD'
            setup_label = f"{setup_tier} {setup_type}"

            # Advisory LLM context from the Sunday weekly analysis.
            # If the stored bias conflicts with the SMC direction, flag it
            # in the alert. Never block a valid setup on LLM alone.
            _smc_direction_str = 'BUY' if poi['direction'] == 'BULLISH' else 'SELL'
            _llm_bias          = self._llm_weekly_bias.get(symbol, '')
            if _llm_bias and _llm_bias not in ('NEUTRAL', ''):
                if _llm_bias == _smc_direction_str:
                    _llm_context = f"Weekly outlook confirms {_llm_bias} bias."
                elif _llm_bias == 'WAIT':
                    _llm_context = (
                        "Weekly outlook advises caution — timeframes mixed. "
                        "Consider reducing size."
                    )
                else:
                    _llm_context = (
                        f"Note: Weekly macro analysis favours {_llm_bias}. "
                        f"This {_smc_direction_str} setup is counter to the weekly bias. "
                        f"Apply extra caution."
                    )
            else:
                _llm_context = ''

            setup_data = {
                'symbol':        symbol,
                'direction':     _smc_direction_str,
                'setup_type':    setup_tier,
                'setup_label':   setup_label,
                'entry_price':   entry_cfg['entry_price'],
                'stop_loss':     sl_cfg['stop_loss'],
                'take_profit_1': tp_cfg['tp1'],
                'take_profit_2': tp_cfg['tp2'],
                'sl_pips':       utils.calculate_pips(
                    symbol, entry_cfg['entry_price'], sl_cfg['stop_loss']),
                'tp1_pips':      tp_cfg.get('tp1_pips', 0.0),
                'tp2_pips':      tp_cfg.get('tp2_pips', 0.0),
                'rr_tp1':        tp_cfg.get('tp1_rr', 0.0),
                'rr_tp2':        tp_cfg.get('tp2_rr', 0.0),
                'ml_score':      consensus,
                'lstm_score':    ml_result['lstm_score'],
                'xgboost_score': ml_result['xgboost_score'],
                'session':       utils.get_session(),
                'timeframe':     'H1',
                'expiry_hours':  8,      # H1 expires in 8 hours; M15=2h; H4=24h
                'order_type':    await self._determine_order_type(
                    symbol,
                    'BUY' if poi['direction'] == 'BULLISH' else 'SELL',
                    entry_cfg['entry_price']
                ),
                'ml_features':   ml_result['features'],
                'news_warning':  news_warn,
                'llm_context':   _llm_context,
                # Chart data is not saved to DB — only used for image generation
                'chart_data': {
                    'm15_df':          m15_df.tail(80),
                    'poi':             poi,
                    'additional_pois': unmitigated[:4],
                    'fvgs':            self.smc.detect_fair_value_gaps(m15_df.tail(60)),
                    'bos_events':      bos_events[:3],
                },
            }

            signal_row = db.save_signal(
                symbol=setup_data['symbol'],
                direction=setup_data['direction'],
                setup_type=setup_data['setup_type'],
                entry_price=setup_data['entry_price'],
                stop_loss=setup_data['stop_loss'],
                take_profit_1=setup_data['take_profit_1'],
                take_profit_2=setup_data['take_profit_2'],
                sl_pips=setup_data['sl_pips'],
                tp1_pips=setup_data['tp1_pips'],
                tp2_pips=setup_data['tp2_pips'],
                rr_tp1=setup_data['rr_tp1'],
                rr_tp2=setup_data['rr_tp2'],
                ml_score=consensus,
                lstm_score=ml_result['lstm_score'],
                xgboost_score=ml_result['xgboost_score'],
                session=setup_data['session'],
                order_type=setup_data.get('order_type', 'LIMIT'),
                timeframe=setup_data.get('timeframe', 'H1'),
                expiry_hours=setup_data.get('expiry_hours', 8),
            )

            if signal_row:
                setup_data['signal_number'] = signal_row.get('signal_number', 0)
                setup_data['signal_id']     = signal_row.get('id')
                auto_execute = self.ml.should_auto_execute(
                    consensus, ml_result['agreement'])
                await self._broadcast_setup(setup_data, auto_execute)
                self.logger.info(
                    "Setup generated and broadcast: %s %s | Score: %d%% | "
                    "Agreement: %s | Tier: %s | Signal #%s",
                    symbol, setup_data['direction'], consensus,
                    ml_result['agreement'], tier, setup_data['signal_number'])

        except Exception as e:
            self.logger.error("Error scanning %s: %s", symbol, e, exc_info=True)

    def _check_filters(
        self, symbol: str, data: pd.DataFrame, htf_trend: dict, poi: dict,
        tier: str = 'STANDARD'
    ) -> bool:
        """
        Run all 5 filters before broadcasting a setup.
        Filter 1: ATR volatility
        Filter 2: Session
        Filter 3: ADX trend strength (was coded in smc_strategy but never called here)
        Filter 4: Premium/discount zone (was coded in smc_strategy but never called here)
        Filter 5: Composite setup quality score
        """
        try:
            direction = poi.get('direction', 'BULLISH')

            # Filter 1: ATR volatility - disabled, news blackout handles spike protection
            atr     = self.smc._calculate_atr(data.tail(20))
            atr_avg = float(data.tail(40)['close'].diff().abs().mean())
            atr_pass, _ = self.smc.check_atr_filter(atr, atr_avg)
            atr_ratio = round(atr / atr_avg, 2) if atr_avg > 0 else 0
            self.logger.debug(
                "%s ATR (informational only, not blocking): ratio=%.2fx - "
                "setup continues regardless.", symbol, atr_ratio)

            # Filter 2: Session - Asian is Unicorn-only, all other sessions trade
            utc_hour     = datetime.now(timezone.utc).hour
            session_name = utils.get_session_name(utc_hour)
            if session_name == 'Asian':
                if tier != 'PREMIUM':
                    self.logger.debug(
                        "%s: Asian session (%d UTC) - only UNICORN setups allowed. "
                        "Current tier: %s. Skipping.", symbol, utc_hour, tier)
                    return False
                self.logger.debug(
                    "%s: Asian session (%d UTC) - UNICORN tier confirmed. Proceeding.",
                    symbol, utc_hour)

            # Filter 3: ADX - reject ranging markets at entry timeframe
            adx_data = data.tail(60)
            if len(adx_data) >= 33:
                pass_adx, adx_val, reason_adx = self.smc.check_adx_filter(adx_data)
                if not pass_adx:
                    self.logger.debug(
                        "%s failed ADX filter: %s (ADX=%.1f)", symbol, reason_adx, adx_val)
                    return False

            # Filter 4: Premium/discount zone - informational and priority tagging only.
            # Zone position does not block a setup. When price is in the correct zone
            # (discount for BUY, premium for SELL) it is flagged as HIGH PRIORITY.
            swing_high    = float(htf_trend.get('swing_high', 0))
            swing_low     = float(htf_trend.get('swing_low',  0))
            current_price = float(data.iloc[-1]['close'])
            if swing_high > swing_low > 0:
                in_zone, zone_pos, zone_desc = self.smc.detect_premium_discount_zone(
                    current_price, swing_high, swing_low, direction)
                correct_zone = (
                    (direction == 'BULLISH' and zone_pos <= 0.45) or
                    (direction == 'BEARISH' and zone_pos >= 0.55)
                )
                if correct_zone:
                    self.logger.info(
                        "%s ZONE - HIGH PRIORITY: %s (zone_pos=%.2f). "
                        "Price is in optimal %s zone.",
                        symbol, zone_desc, zone_pos,
                        'discount' if direction == 'BULLISH' else 'premium')
                else:
                    self.logger.info(
                        "%s ZONE - STANDARD: %s (zone_pos=%.2f). "
                        "Not in optimal zone but setup proceeds on other merits.",
                        symbol, zone_desc, zone_pos)

            # Filter 5: Composite setup quality score
            try:
                bos_events = self.smc.detect_break_of_structure(
                    data.tail(50), htf_trend.get('trend', 'BULLISH'))
                quality = self.smc.score_setup_quality(
                    data, poi, htf_trend, bos_events, direction, symbol)
                self.logger.debug(
                    "%s quality score: %d (informational only, not blocking).",
                    symbol, quality)
            except Exception as qe:
                self.logger.warning(
                    "Quality score failed for %s (%s). Continuing without it.", symbol, qe)

            return True

        except Exception as e:
            self.logger.error("Filter check error for %s: %s", symbol, e)
            return False
        
    async def _determine_order_type(
        self, symbol: str, direction: str, entry_price: float
    ) -> str:
        """
        Determine correct order type by comparing current market price
        to the calculated entry price.

        For BUY:
          - Current ask <= entry: price is already at or below entry -> MARKET
          - Current ask > entry:  price needs to pull back to entry -> LIMIT
        For SELL:
          - Current bid >= entry: price is already at or above entry -> MARKET
          - Current bid < entry:  price needs to rally to entry -> LIMIT

        Falls back to LIMIT on any error to ensure the order is placed
        as a pending order rather than executing at a worse price.
        """
        try:
            bid, ask = await self.mt5.get_current_price(symbol)
            if bid is None or ask is None:
                return 'LIMIT'

            if direction == 'BUY':
                return 'MARKET' if ask <= entry_price else 'LIMIT'
            else:
                return 'MARKET' if bid >= entry_price else 'LIMIT'

        except Exception as e:
            self.logger.debug(
                "Order type determination failed for %s, defaulting to LIMIT: %s",
                symbol, e)
            return 'LIMIT'

    # ==================== BROADCAST ====================

    async def _broadcast_setup(self, setup_data: dict, auto_execute: bool):
        """
        Send the setup alert to all subscribed users.
        Users with MT5 connected are auto-executed if score meets threshold.
        Rate-limited to stay within Telegram's 30 messages/second limit.
        """
        try:
            subscribed = db.get_subscribed_users()
            if not subscribed:
                return

            signal_num = setup_data.get('signal_number', 0)
            news_warn  = setup_data.get('news_warning', '')

            # Fetch live M15 data for chart generation.
            # Priority: MetaApi -> MT5 worker -> skip image entirely.
            # If neither source delivers data, the broadcast continues as
            # text-only with no error raised to the user.
            chart_bytes = None
            chart_data  = setup_data.get('chart_data')
            _symbol     = setup_data.get('symbol', '')

            if chart_data is not None and self._chart_gen is not None:
                _live_raw = None
                try:
                    _live_raw = await self.mt5.get_historical_data(
                        _symbol, 'M15', bars=100)
                except Exception as _live_err:
                    self.logger.debug(
                        "Live M15 fetch for chart failed (%s): %s",
                        _symbol, _live_err)

                if _live_raw and len(_live_raw) >= 20:
                    try:
                        _live_df   = self._candles_to_df(_live_raw)
                        _loop      = asyncio.get_running_loop()
                        # Capture loop-local variables before passing to executor
                        _poi       = chart_data.get('poi')
                        _add_pois  = chart_data.get('additional_pois', [])
                        _fvgs      = chart_data.get('fvgs', [])
                        _bos       = chart_data.get('bos_events', [])
                        _sd        = dict(setup_data)   # shallow copy is safe here
                        _df_snap   = _live_df.tail(80).copy()
                        chart_bytes = await _loop.run_in_executor(
                            None,
                            lambda: self._chart_gen.generate_setup_chart(
                                data=_df_snap,
                                setup_data=_sd,
                                poi=_poi,
                                additional_pois=_add_pois,
                                fvgs=_fvgs,
                                bos_events=_bos,
                            )
                        )
                        if chart_bytes:
                            self.logger.info(
                                "Chart generated for setup #%d with live data "
                                "(%d KB).",
                                signal_num, len(chart_bytes) // 1024)
                        else:
                            self.logger.info(
                                "Chart generator returned None for #%d. "
                                "Text-only broadcast.", signal_num)
                    except Exception as _chart_err:
                        self.logger.warning(
                            "Chart generation failed for setup #%d: %s",
                            signal_num, _chart_err)
                else:
                    self.logger.info(
                        "Insufficient live M15 data for %s chart. "
                        "Text-only broadcast.", _symbol)

            sent = 0
            for user in subscribed:
                tid = user['telegram_id']
                try:
                    # Calculate lot size per user's risk setting if MT5 connected.
                    # mt5_connected flag checked first, then credentials as fallback
                    # so a user who connected but whose flag was not set still executes.
                    lot_size = None
                    mt5_flag    = bool(user.get('mt5_connected'))
                    can_execute = mt5_flag

                    if not auto_execute:
                        self.logger.info(
                            "AUTO-EXECUTE SKIP | user %d | reason: auto_execute=False "
                            "(consensus below threshold or agreement=WEAK)", tid)
                    elif not can_execute:
                        self.logger.info(
                            "AUTO-EXECUTE SKIP | user %d | reason: mt5_connected=%s. "
                            "User has not linked MT5 account.",
                            tid, mt5_flag)
                    else:
                        lot_size = await self._calculate_user_lot_size(user, setup_data)
                        if lot_size is None:
                            self.logger.info(
                                "AUTO-EXECUTE SKIP | user %d | reason: lot_size "
                                "calculation returned None (check account balance "
                                "and MT5 worker connection).", tid)
                        else:
                            self.logger.info(
                                "AUTO-EXECUTE READY | user %d | lot_size=%.2f",
                                tid, lot_size)

                    message = utils.format_setup_message(
                        signal_number=signal_num,
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
                        ml_score=setup_data['ml_score'],
                        session=setup_data.get('session', 'N/A'),
                        order_type=setup_data.get('order_type', 'LIMIT'),
                        lot_size=lot_size,
                        expiry_hours=int(setup_data.get('expiry_hours', 8)),
                    )

                    if news_warn:
                        message += f"\n\n{news_warn}"
                    _llm_ctx = setup_data.get('llm_context', '')
                    if _llm_ctx:
                        message += f"\n\n{_llm_ctx}"

                    try:
                        # Chart images are a Basic+ feature.
                        # Free-tier subscribers receive text-only alerts.
                        from payment_handler import get_subscription_manager as _gsm
                        _user_tier = _gsm().get_tier(tid)
                        if chart_bytes and _user_tier in ('basic', 'pro', 'admin'):
                            try:
                                await self.bot.send_photo(
                                    chat_id=tid,
                                    photo=InputFile(
                                        io.BytesIO(chart_bytes),
                                        filename=f'setup_{signal_num}.png',
                                    ),
                                )
                            except Exception as photo_err:
                                self.logger.warning(
                                    "Chart photo failed for user %d: %s. "
                                    "Sending text only.", tid, photo_err)

                        await self.bot.send_message(chat_id=tid, text=message)
                        sent += 1
                    except Forbidden:
                        self.logger.info("User %d blocked the bot. Skipping.", tid)
                    except NetworkError as ne:
                        self.logger.warning(
                            "Network error sending setup to user %d: %s. "
                            "Queuing message.", tid, ne)
                        db.queue_message(tid, message, 'SETUP_ALERT')
                    except Exception as send_err:
                        self.logger.error(
                            "Error sending setup to user %d: %s", tid, send_err)
                        db.queue_message(tid, message, 'SETUP_ALERT')

                    # Auto-execute trade for connected users
                    if auto_execute and can_execute and lot_size:
                        await self._auto_execute_trade(user, setup_data, lot_size)

                except Exception as e:
                    self.logger.error(
                        "Error processing user %d in broadcast: %s", tid, e)
                finally:
                    await asyncio.sleep(_TELEGRAM_SEND_DELAY)

            self.logger.info(
                "Broadcast complete: %d of %d users received setup %s %s.",
                sent, len(subscribed),
                setup_data['symbol'], setup_data['direction'])

        except Exception as e:
            self.logger.error("Broadcast error: %s", e, exc_info=True)

    # ==================== AUTO-EXECUTION ====================

    async def _calculate_user_lot_size(
        self, user: dict, setup_data: dict
    ) -> Optional[float]:
        """
        Calculate the appropriate lot size for this user's risk settings
        and account balance. Returns None if calculation fails.
        """
        try:
            creds = db.get_mt5_credentials(user['telegram_id'])
            if not creds:
                return None

            ok, account_info = await self.mt5.get_account_info(user['telegram_id'])
            if not ok or not account_info:
                return None

            balance  = float(account_info.get('balance', 0))
            currency = account_info.get('currency', 'USD')
            risk_pct = float(user.get('risk_percent', config.DEFAULT_RISK_PERCENT))
            sl_pips  = float(setup_data.get('sl_pips', 20.0))

            if balance <= 0 or sl_pips <= 0:
                return None

            exchange_rates = await self.mt5.get_exchange_rates()
            lot_size = utils.calculate_lot_size(
                balance, risk_pct, sl_pips,
                setup_data['symbol'], currency, exchange_rates)
            return lot_size

        except Exception as e:
            self.logger.error(
                "Lot size calculation failed for user %d: %s",
                user['telegram_id'], e)
            return None

    async def _auto_execute_trade(
        self, user: dict, setup_data: dict, lot_size: float
    ):
        """
        Place a trade on behalf of the user via the MT5 worker.
        On success, register the trade with the position monitor for
        TP1/TP2 automation and ML outcome recording.
        """
        tid = user['telegram_id']
        try:
            symbol    = setup_data['symbol']
            direction = setup_data['direction']

            # Check daily loss limit before executing
            day_start_utc = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            daily_loss = db.get_daily_loss_percent(tid, since=day_start_utc)
            if daily_loss is not None and daily_loss >= config.MAX_DAILY_LOSS_PERCENT:
                self.logger.warning(
                    "User %d has reached daily loss limit (%.1f%%). "
                    "Skipping auto-execution.", tid, daily_loss)
                return

            # Determine order type (LIMIT vs MARKET) based on current price
            bid, ask = await self.mt5.get_current_price(symbol)
            current = ask if direction == 'BUY' else bid
            entry   = setup_data['entry_price']

            if direction == 'BUY':
                order_type = 'MARKET' if (current is not None and current <= entry) else 'LIMIT'
            else:
                order_type = 'MARKET' if (current is not None and current >= entry) else 'LIMIT'

            expiry_minutes = int(setup_data.get('expiry_hours', 8)) * 60

            success, ticket, actual_lot, message = await self.mt5.place_order(
                telegram_id=tid,
                symbol=symbol,
                direction=direction,
                lot_size=lot_size,
                entry_price=entry,
                stop_loss=setup_data['stop_loss'],
                take_profit=setup_data['take_profit_1'],
                take_profit_2=setup_data['take_profit_2'],
                order_type=order_type,
                comment='NIXIE TRADES',
                sl_pips=float(setup_data.get('sl_pips', 20.0)),
                risk_percent=float(user.get('risk_percent', config.DEFAULT_RISK_PERCENT)),
                expiry_minutes=expiry_minutes,
            )

            if success and ticket:
                self.logger.info(
                    "Auto-executed trade for user %d: %s %s %.2f lots "
                    "at %.5f (ticket %d).",
                    tid, direction, symbol, actual_lot, entry, ticket)

                # Register with position monitor for TP1/TP2 automation
                if self.monitor is not None:
                    # Fetch account currency for accurate P&L display.
                    _acct_ccy = 'USD'
                    try:
                        _ok2, _acct2 = self.mt5.get_account_info(tid)
                        if _ok2 and _acct2:
                            _acct_ccy = _acct2.get('currency', 'USD')
                    except Exception:
                        pass

                    self.monitor.add_position(
                        ticket=ticket,
                        symbol=symbol,
                        direction=direction,
                        volume=actual_lot,
                        entry_price=entry,
                        stop_loss=setup_data['stop_loss'],
                        take_profit_1=setup_data['take_profit_1'],
                        take_profit_2=setup_data['take_profit_2'],
                        telegram_id=tid,
                        magic_number=config.MAGIC_NUMBER,
                        account_currency=_acct_ccy,
                        ml_features=setup_data.get('ml_features'),
                    )

                # Save trade record to database
                db.save_trade(
                    telegram_id=tid,
                    signal_id=setup_data.get('signal_id'),
                    symbol=symbol,
                    direction=direction,
                    lot_size=actual_lot,
                    entry_price=entry,
                    fill_price=entry,
                    stop_loss=setup_data['stop_loss'],
                    take_profit_1=setup_data['take_profit_1'],
                    take_profit_2=setup_data['take_profit_2'],
                    order_type=order_type,
                    ticket=ticket,
                )
            else:
                self.logger.error(
                    "Auto-execution failed for user %d (%s %s): %s",
                    tid, direction, symbol, message)

        except Exception as e:
            self.logger.error(
                "Auto-execution error for user %d: %s", tid, e, exc_info=True)

    # ==================== MARKET OVERVIEW ====================

    async def _generate_market_overview(self) -> str:
        """
        Build the market structure section of the daily 8 AM alert.
        Analyses D1 trend for each major pair.
        """
        try:
            if not await self.mt5.is_worker_reachable():
                return "Market data unavailable (MetaApi not connected)."

            lines = ["MARKET STRUCTURE:"]
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']

            for symbol in major_pairs:
                try:
                    raw = await self.mt5.get_historical_data(symbol, 'D1', bars=50)
                    if not raw:
                        continue
                    df = self._candles_to_df(raw)

                    htf = self.smc.determine_htf_trend(df)
                    trend = htf.get('trend', 'UNKNOWN')
                    conf  = htf.get('confidence', 0)
                    lines.append(f"  {symbol}: {trend} ({conf}% confidence)")
                except Exception as e:
                    self.logger.warning(
                        "Could not analyse %s for market overview: %s", symbol, e)

            return "\n".join(lines) if len(lines) > 1 else "Market data unavailable."

        except Exception as e:
            self.logger.error("Error generating market overview: %s", e)
            return "Market overview unavailable."
        
    # ==================== PER-USER TIMED MESSAGE DISPATCHER ====================

    async def _timed_messages(self):
        """
        Runs on the configured alert-check interval. Checks each subscribed user's local time
        and sends the appropriate scheduled message if it is due.

        Delivery schedule (in each user's own timezone):
          06:30  Daily Market Briefing
          08:00  Red Folder News Alert
          Sunday 09:00  Weekly Market Analysis
        """
        try:
            import pytz
            users = db.get_subscribed_users()
            if not users:
                return

            utc_now  = datetime.now(timezone.utc)
            date_key = utc_now.strftime('%Y-%m-%d')

            # Reset tracking sets each new UTC day
            if date_key not in self._briefing_sent:
                self._briefing_sent = {date_key: set()}
                self._news_sent     = {date_key: set()}
                self._weekly_sent   = {date_key: set()}

            for user in users:
                tid    = user['telegram_id']
                tz_str = user.get('user_timezone') or user.get('timezone') or 'UTC'

                try:
                    user_tz = pytz.timezone(tz_str)
                except Exception:
                    user_tz = pytz.UTC

                user_now  = utc_now.astimezone(user_tz)
                weekday   = user_now.weekday()   # 0=Monday, 6=Sunday

                def _is_due(target_hour: int, target_minute: int) -> bool:
                    due_at = user_now.replace(
                        hour=target_hour,
                        minute=target_minute,
                        second=0,
                        microsecond=0,
                    )
                    minutes_since = (user_now - due_at).total_seconds() / 60.0
                    return 0 <= minutes_since < self.alert_check_interval_minutes

                # 06:30 Daily Market Briefing
                if _is_due(6, 30):
                    if tid not in self._briefing_sent[date_key]:
                        self._briefing_sent[date_key].add(tid)
                        asyncio.create_task(
                            self._send_daily_briefing(tid, user_now)
                        )

                # 08:00 Red Folder News Alert
                if _is_due(8, 0):
                    if tid not in self._news_sent[date_key]:
                        self._news_sent[date_key].add(tid)
                        asyncio.create_task(
                            self._send_news_alert(tid, user_now)
                        )

                # Sunday 09:00 Weekly Analysis
                if weekday == 6 and _is_due(9, 0):
                    if tid not in self._weekly_sent[date_key]:
                        self._weekly_sent[date_key].add(tid)
                        asyncio.create_task(
                            self._send_weekly_analysis(tid, user_now)
                        )

        except Exception as e:
            self.logger.error("Error in _timed_messages: %s", e, exc_info=True)

    # ==================== 06:30 DAILY MARKET BRIEFING ====================

    async def _send_daily_briefing(self, telegram_id: int, user_now: datetime):
        """
        06:30 AM per-user timezone: market structure summary for the day ahead.
        Shows HTF trend and confidence for all major pairs.
        Basic and higher subscribers only.
        """
        try:
            from payment_handler import get_subscription_manager as _gsm
            if _gsm().get_tier(telegram_id) == 'free':
                return   # Daily briefing is a Basic feature — skip silently

            date_str = user_now.strftime('%A, %B %d, %Y')
            session  = utils.get_session()

            lines = [
                "DAILY MARKET BRIEFING",
                date_str,
                "",
                "Good morning. Here is your pre-session market structure summary.",
                "",
                "MARKET STRUCTURE (Daily Timeframe):",
                "",
            ]

            try:
                mt5_ok = await self.mt5.is_worker_reachable()
            except Exception:
                mt5_ok = False

            if mt5_ok:
                for symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'GBPJPY', 'AUDUSD']:
                    try:
                        raw = await self.mt5.get_historical_data(symbol, 'D1', bars=100)
                        if not raw:
                            continue
                        df = self._candles_to_df(raw)
                        htf        = self.smc.determine_htf_trend(df)
                        trend      = htf.get('trend', 'UNKNOWN')
                        confidence = htf.get('confidence', 0)
                        lines.append(f"  {symbol:<10} {trend:<8}  ({confidence}% confidence)")
                    except Exception:
                        lines.append(f"  {symbol:<10} Data unavailable")
            else:
                lines.append("  Market data unavailable. MT5 worker is offline.")

            # Convert session windows from UTC to user's local time
            try:
                import pytz
                user_tz   = user_now.tzinfo
                ref_date  = user_now.date()

                def _utc_to_local(h: int, m: int = 0) -> str:
                    utc_dt = datetime(ref_date.year, ref_date.month, ref_date.day,
                                      h, m, tzinfo=pytz.utc)
                    local  = utc_dt.astimezone(user_tz)
                    return local.strftime('%I:%M %p').lstrip('0')

                london_open  = _utc_to_local(7)
                london_close = _utc_to_local(16)
                ny_open      = _utc_to_local(13)
                ny_close     = _utc_to_local(22)
                ov_open      = _utc_to_local(13)
                ov_close     = _utc_to_local(16)
            except Exception:
                london_open  = '07:00 AM UTC'
                london_close = '04:00 PM UTC'
                ny_open      = '01:00 PM UTC'
                ny_close     = '10:00 PM UTC'
                ov_open      = '01:00 PM UTC'
                ov_close     = '04:00 PM UTC'

            lines += [
                "",
                f"Current Session: {session}",
                "",
                "Active trading sessions (your local time):",
                f"  London:   {london_open} - {london_close}",
                f"  New York: {ny_open} - {ny_close}",
                f"  Overlap:  {ov_open} - {ov_close} (highest volume)",
                "",
                f"The system scans all pairs every {self.market_scan_interval_minutes} minutes.",
                "Qualifying setups will be sent automatically when detected.",
                "",
                config.FOOTER,
            ]

            message = utils.validate_user_message("\n".join(lines))
            await self._safe_send(telegram_id, message)

        except Exception as e:
            self.logger.error("Error in _send_daily_briefing for user %d: %s", telegram_id, e)

    # ==================== 08:00 RED FOLDER NEWS ALERT ====================

    async def _send_news_alert(self, telegram_id: int, user_now: datetime):
        """
        08:00 AM per-user timezone: high-impact news events for today.
        Falls back gracefully if live news fetch fails.
        Basic and higher subscribers only.
        """
        try:
            from payment_handler import get_subscription_manager as _gsm
            if _gsm().get_tier(telegram_id) == 'free':
                return   # News alerts are a Basic feature — skip silently

            date_str = user_now.strftime('%A, %B %d, %Y')

            try:
                events = self.news.get_red_folder_events(hours_ahead=24)
                if events:
                    event_lines = []
                    user_tz = user_now.tzinfo
                    for ev in events:
                        try:
                            # Convert UTC event time to user's local timezone
                            ev_utc  = ev.timestamp
                            if ev_utc.tzinfo is None:
                                import pytz
                                ev_utc = pytz.utc.localize(ev_utc)
                            ev_local   = ev_utc.astimezone(user_tz)
                            day_str    = ev_local.strftime('%a')
                            time_str   = ev_local.strftime('%I:%M %p').lstrip('0')
                            local_str  = f"{day_str} {time_str} (your time)"
                        except Exception:
                            local_str = 'Unknown time'
                        currency = getattr(ev, 'currency', 'N/A')
                        title    = getattr(ev, 'title', str(ev))
                        event_lines.append(f"  {local_str}  {currency:<4}  {title}")
                    news_body = "\n".join(event_lines)
                else:
                    news_body = "  No high-impact events found for today."
            except Exception:
                news_body = (
                    "  Live news feed unavailable.\n"
                    "  Check Forex Factory for today's events:\n"
                    "  https://www.forexfactory.com/calendar"
                )

            lines = [
                "HIGH-IMPACT NEWS ALERT",
                date_str,
                "",
                "Red folder events scheduled today.",
                "Trading is paused automatically 30 minutes before",
                "and 15 minutes after each high-impact event.",
                "",
                "RED FOLDER EVENTS:",
                "",
                news_body,
                "",
                "Plan your trades around these windows.",
                "",
                config.FOOTER,
            ]

            message = utils.validate_user_message("\n".join(lines))
            await self._safe_send(telegram_id, message)

        except Exception as e:
            self.logger.error("Error in _send_news_alert for user %d: %s", telegram_id, e)

    # ==================== SUNDAY 09:00 WEEKLY ANALYSIS ====================

    async def _send_weekly_analysis(self, telegram_id: int, user_now: datetime):
        """
        Sunday 09:00 AM per-user timezone: weekly bias for all pairs
        plus key news events for the coming week.
        Includes an LLM-generated macro and sentiment overview.
        Pro and Admin subscribers only.
        """
        try:
            from payment_handler import get_subscription_manager as _gsm
            if _gsm().get_tier(telegram_id) not in ('pro', 'admin'):
                return

            date_str = user_now.strftime('%A, %B %d, %Y')

            # ── Collect technical analysis for all pairs ─────────────────
            _analysis_symbols = [
                'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
                'USDCAD', 'NZDUSD', 'XAUUSD', 'GBPJPY', 'EURJPY',
            ]
            pair_results = {}

            try:
                mt5_ok = await self.mt5.is_worker_reachable()
            except Exception:
                mt5_ok = False

            if mt5_ok:
                for symbol in _analysis_symbols:
                    try:
                        raw_mn1 = await self.mt5.get_historical_data(
                            symbol, 'MN1', bars=24)
                        raw_w1 = await self.mt5.get_historical_data(
                            symbol, 'W1', bars=52)
                        raw_d1 = await self.mt5.get_historical_data(
                            symbol, 'D1', bars=300)

                        if not raw_d1:
                            pair_results[symbol] = {
                                'mn1': 'N/A', 'w1': 'N/A',
                                'd1': 'UNAVAILABLE', 'd1_conf': 0,
                                'alignment': 'UNAVAILABLE',
                                'bias': 'NEUTRAL',
                            }
                            continue

                        d1_df  = self._candles_to_df(raw_d1)
                        d1_htf = self.smc.determine_htf_trend(d1_df)
                        d1_trend = d1_htf.get('trend', 'UNKNOWN')
                        d1_conf  = int(d1_htf.get('confidence', 0))

                        mn1_trend = 'N/A'
                        w1_trend  = 'N/A'

                        if raw_mn1 and len(raw_mn1) >= 10:
                            mn1_df  = self._candles_to_df(raw_mn1)
                            mn1_htf = self.smc.determine_htf_trend(mn1_df)
                            mn1_trend = mn1_htf.get('trend', 'UNKNOWN')

                        if raw_w1 and len(raw_w1) >= 10:
                            w1_df  = self._candles_to_df(raw_w1)
                            w1_htf = self.smc.determine_htf_trend(w1_df)
                            w1_trend = w1_htf.get('trend', 'UNKNOWN')

                        known = [
                            t for t in [mn1_trend, w1_trend, d1_trend]
                            if t not in ('UNKNOWN', 'N/A', 'RANGING')
                        ]
                        if len(known) >= 2 and len(set(known)) == 1:
                            alignment = 'ALIGNED'
                        elif len(known) >= 2 and len(set(known)) > 1:
                            alignment = 'MIXED'
                        else:
                            alignment = 'RANGING'

                        # Derive a plain-English bias from alignment + D1
                        if alignment == 'ALIGNED' and d1_trend == 'BULLISH':
                            bias = 'BUY'
                        elif alignment == 'ALIGNED' and d1_trend == 'BEARISH':
                            bias = 'SELL'
                        elif alignment == 'MIXED':
                            bias = 'WAIT'
                        else:
                            bias = 'NEUTRAL'

                        pair_results[symbol] = {
                            'mn1':       mn1_trend,
                            'w1':        w1_trend,
                            'd1':        d1_trend,
                            'd1_conf':   d1_conf,
                            'alignment': alignment,
                            'bias':      bias,
                        }

                    except Exception as _sym_err:
                        self.logger.warning(
                            "Weekly analysis failed for %s: %s",
                            symbol, _sym_err)
                        pair_results[symbol] = {
                            'mn1': 'N/A', 'w1': 'N/A',
                            'd1': 'ERROR', 'd1_conf': 0,
                            'alignment': 'UNAVAILABLE',
                            'bias': 'NEUTRAL',
                        }

            # ── Fetch upcoming news ───────────────────────────────────────
            news_lines = []
            try:
                events = self.news.get_red_folder_events(hours_ahead=168)
                user_tz = user_now.tzinfo
                import pytz
                for ev in (events or [])[:8]:
                    try:
                        ev_utc = ev.timestamp
                        if ev_utc.tzinfo is None:
                            ev_utc = pytz.utc.localize(ev_utc)
                        ev_local  = ev_utc.astimezone(user_tz)
                        day_str   = ev_local.strftime('%a')
                        time_str  = ev_local.strftime('%I:%M %p').lstrip('0')
                        currency  = getattr(ev, 'currency', 'N/A')
                        title     = getattr(ev, 'title', '')
                        news_lines.append(
                            f"  {day_str} {time_str:<8}  {currency:<4}  {title}"
                        )
                    except Exception:
                        pass
            except Exception:
                pass

            # ── LLM macro + sentiment + fundamental analysis ──────────────
            llm_overview    = ''
            llm_trade_plan  = ''
            try:
                llm_overview, llm_trade_plan = await self._get_llm_weekly_analysis(
                    pair_results, news_lines, date_str)
            except Exception as _llm_err:
                self.logger.warning(
                    "LLM weekly analysis failed for user %d: %s",
                    telegram_id, _llm_err)

            # Store the SMC bias from pair_results for use in daily scans.
            # This is the advisory layer: LLM does not override SMC/ML,
            # it provides a confirmation or conflict flag on each setup.
            if pair_results:
                for _sym, _r in pair_results.items():
                    self._llm_weekly_bias[_sym] = _r.get('bias', 'NEUTRAL')
                self.logger.info(
                    "Weekly LLM bias stored for %d symbols.",
                    len(pair_results))

            # ── Build message ─────────────────────────────────────────────
            lines = [
                "WEEKLY MARKET ANALYSIS",
                date_str,
                "",
            ]

            # Macro overview (LLM block)
            if llm_overview:
                lines += [
                    "MACRO AND SENTIMENT OVERVIEW",
                    "",
                    llm_overview,
                    "",
                ]

            # Technical bias per pair
            if pair_results:
                lines += [
                    "PAIR BIAS THIS WEEK",
                    "(Monthly | Weekly | Daily — highest confidence timeframe wins)",
                    "",
                ]

                # Bias icons for readability
                _bias_icon = {
                    'BUY':     'LONG  ',
                    'SELL':    'SHORT ',
                    'WAIT':    'WAIT  ',
                    'NEUTRAL': 'NEUTRAL',
                }
                _align_icon = {
                    'ALIGNED':     'ALIGNED',
                    'MIXED':       'MIXED  ',
                    'RANGING':     'RANGING',
                    'UNAVAILABLE': 'N/A    ',
                }

                for symbol, r in pair_results.items():
                    bias_str  = _bias_icon.get(r['bias'], 'NEUTRAL')
                    align_str = _align_icon.get(r['alignment'], 'N/A    ')
                    mn1 = r['mn1'][:4] if r['mn1'] not in ('N/A', 'UNKNOWN') else '----'
                    w1  = r['w1'][:4]  if r['w1']  not in ('N/A', 'UNKNOWN') else '----'
                    d1  = r['d1'][:4]  if r['d1']  not in ('N/A', 'UNKNOWN', 'ERROR', 'UNAVAILABLE') else '----'
                    conf = r['d1_conf']

                    lines.append(
                        f"{symbol:<8}  [{bias_str}]  {align_str}"
                    )
                    lines.append(
                        f"         MN1: {mn1:<4}  W1: {w1:<4}  D1: {d1:<4} ({conf}%)"
                    )
                    lines.append("")

                lines += [
                    "BIAS KEY",
                    "  LONG    - Aligned bullish. Look for BUY setups on demand zones.",
                    "  SHORT   - Aligned bearish. Look for SELL setups on supply zones.",
                    "  WAIT    - Timeframes conflict. Reduce size or wait for clarity.",
                    "  NEUTRAL - No clear direction. Avoid new positions.",
                    "",
                ]

            # High-impact news this week
            if news_lines:
                lines += [
                    "HIGH-IMPACT NEWS THIS WEEK",
                    "(your local time)",
                    "",
                ]
                lines.extend(news_lines)
                lines.append("")

            # LLM trade plan
            if llm_trade_plan:
                lines += [
                    "WEEKLY TRADE PLAN",
                    "",
                    llm_trade_plan,
                    "",
                ]

            lines += [
                "TRADING GUIDELINES",
                "",
                "  - Only trade pairs with a clear LONG or SHORT bias",
                "  - Avoid entries 30 min before and 15 min after red events",
                "  - Best sessions: London Open (07:00 UTC) and New York Open (13:00 UTC)",
                "  - WAIT and NEUTRAL pairs: stand aside until structure clarifies",
                "",
                "EDUCATIONAL PURPOSES ONLY. NOT FINANCIAL ADVICE.",
                "Past performance is not indicative of future results.",
                "",
                config.FOOTER,
            ]

            message = utils.validate_user_message("\n".join(lines))
            await self._safe_send(telegram_id, message)

        except Exception as e:
            self.logger.error(
                "Error in _send_weekly_analysis for user %d: %s",
                telegram_id, e, exc_info=True)

    async def _get_llm_weekly_analysis(
        self,
        pair_results: dict,
        news_lines: list,
        date_str: str,
    ) -> tuple:
        """
        Call Google Gemini (free tier) to generate a macro + sentiment +
        fundamental overview and a weekly trade plan grounded in the
        SMC technical data already collected.

        Model: gemini-2.0-flash (free, 15 req/min, 1M tokens/day)
        API key: config.GOOGLE_API_KEY from Google AI Studio (free)

        Returns:
            (overview_text: str, trade_plan_text: str)
            Returns ('', '') on any failure — technical data still sends.
        """
        api_key = getattr(config, 'GOOGLE_API_KEY', '').strip()
        if not api_key:
            self.logger.info(
                "GOOGLE_API_KEY not set. Skipping LLM analysis. "
                "Add it to .env to enable macro commentary.")
            return '', ''

        try:
            import httpx

            # Build technical context string for the model
            tech_lines = []
            for symbol, r in pair_results.items():
                tech_lines.append(
                    f"{symbol}: Monthly={r['mn1']} Weekly={r['w1']} "
                    f"Daily={r['d1']} ({r['d1_conf']}% confidence) "
                    f"SMC_Bias={r['bias']} Alignment={r['alignment']}"
                )
            tech_summary = "\n".join(tech_lines) if tech_lines else "No data."
            news_summary = (
                "\n".join(news_lines[:8]) if news_lines
                else "No high-impact events found this week."
            )
            daily_ctx = self._last_daily_context[:600] if self._last_daily_context else ''

            system_prompt = (
                "You are an institutional forex and commodity market analyst "
                "for Nixie Trades, an algorithmic trading education platform "
                "that uses Smart Money Concepts (SMC). "
                "You write concise, professional, factual commentary that "
                "combines macro fundamentals, central bank policy, market "
                "sentiment, and SMC technical structure. "
                "You never guarantee returns, never give direct buy/sell "
                "instructions as financial advice, and always frame analysis "
                "as educational content. "
                "No markdown. No bullet symbols. No asterisks. "
                "Short paragraphs only. Plain English suitable for Telegram. "
                "Maximum 120 words per section."
            )

            user_prompt = (
                f"Today is {date_str}.\n\n"
                f"SMC TECHNICAL STRUCTURE (multi-timeframe):\n{tech_summary}\n\n"
                f"HIGH-IMPACT ECONOMIC EVENTS THIS WEEK:\n{news_summary}\n\n"
                + (f"RECENT DAILY BRIEFING CONTEXT:\n{daily_ctx}\n\n"
                   if daily_ctx else "")
                + "Provide TWO sections:\n\n"
                "SECTION 1 - MACRO AND SENTIMENT (max 110 words):\n"
                "Describe the macro and fundamental backdrop this week. "
                "Cover central bank tone (Fed, ECB, BOE, BOJ), risk sentiment "
                "(risk-on vs risk-off), USD strength or weakness, and commodity "
                "drivers for Gold. Explain how the scheduled news events "
                "could shift institutional order flow. Be specific.\n\n"
                "SECTION 2 - WEEKLY TRADE PLAN (max 110 words):\n"
                "Based on SMC alignment and the macro backdrop, describe "
                "which pairs offer the highest-probability setups, which "
                "to avoid, and why. Mention session timing (London, New York). "
                "Do not give specific entry prices. "
                "Frame as educational analysis only, not financial advice.\n\n"
                "Format your response EXACTLY as:\n"
                "MACRO_OVERVIEW: [section 1 text here]\n"
                "TRADE_PLAN: [section 2 text here]"
            )

            url     = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.0-flash:generateContent?key={api_key}"
            )
            payload = {
                "system_instruction": {
                    "parts": [{"text": system_prompt}]
                },
                "contents": [
                    {
                        "role":  "user",
                        "parts": [{"text": user_prompt}],
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": 600,
                    "temperature":     0.30,
                    "topP":            0.90,
                },
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

            if resp.status_code != 200:
                self.logger.warning(
                    "Gemini API returned %d: %s",
                    resp.status_code, resp.text[:300])
                return '', ''

            data      = resp.json()
            raw_text  = ''
            try:
                raw_text = (
                    data['candidates'][0]['content']['parts'][0]['text']
                )
            except (KeyError, IndexError, TypeError) as parse_err:
                self.logger.warning(
                    "Gemini response parse error: %s | raw: %s",
                    parse_err, str(data)[:300])
                return '', ''

            # Parse the two named sections
            overview   = ''
            trade_plan = ''

            if 'MACRO_OVERVIEW:' in raw_text and 'TRADE_PLAN:' in raw_text:
                parts      = raw_text.split('TRADE_PLAN:', 1)
                overview   = parts[0].replace('MACRO_OVERVIEW:', '').strip()
                trade_plan = parts[1].strip()
            elif 'MACRO_OVERVIEW:' in raw_text:
                overview = raw_text.replace('MACRO_OVERVIEW:', '').strip()
            else:
                overview = raw_text.strip()

            # Strip any markdown the model may emit despite instructions
            for ch in ('**', '*', '`', '##', '#'):
                overview   = overview.replace(ch, '')
                trade_plan = trade_plan.replace(ch, '')

            overview   = overview.strip()
            trade_plan = trade_plan.strip()

            self.logger.info(
                "Gemini weekly analysis generated: overview=%d chars, "
                "trade_plan=%d chars.",
                len(overview), len(trade_plan))
            return overview, trade_plan

        except ImportError:
            self.logger.warning(
                "httpx not installed. Run: "
                "pip install httpx --break-system-packages")
            return '', ''
        except Exception as exc:
            self.logger.warning("Gemini API error: %s", exc)
            return '', ''
        try:
            from payment_handler import get_subscription_manager as _gsm
            if _gsm().get_tier(telegram_id) not in ('pro', 'admin'):
                return   # Weekly analysis is a Pro feature — skip silently

            date_str = user_now.strftime('%A, %B %d, %Y')

            lines = [
                "WEEKLY MARKET ANALYSIS",
                date_str,
                "",
                "Good morning. Here is your weekly outlook for the week ahead.",
                "",
            ]

            try:
                mt5_ok = await self.mt5.is_worker_reachable()
            except Exception:
                mt5_ok = False

            if mt5_ok:
                lines += [
                    "TOP-DOWN MARKET STRUCTURE ANALYSIS",
                    "",
                    "Reading: Monthly sets the macro bias. Weekly confirms the intermediate",
                    "structure. Daily identifies the entry-level trend. A setup is highest",
                    "quality when all three timeframes agree on direction.",
                    "",
                ]
                _analysis_symbols = [
                    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
                    'USDCAD', 'NZDUSD', 'XAUUSD', 'GBPJPY', 'EURJPY',
                ]
                for symbol in _analysis_symbols:
                    try:
                        # Fetch all three timeframes
                        raw_mn1 = await self.mt5.get_historical_data(symbol, 'MN1', bars=24)
                        raw_w1  = await self.mt5.get_historical_data(symbol, 'W1',  bars=52)
                        raw_d1  = await self.mt5.get_historical_data(symbol, 'D1',  bars=300)

                        if not raw_d1:
                            lines.append(f"  {symbol:<10} Data unavailable")
                            continue

                        d1_df = self._candles_to_df(raw_d1)
                        d1_htf = self.smc.determine_htf_trend(d1_df)
                        d1_trend = d1_htf.get('trend', 'UNKNOWN')
                        d1_conf  = d1_htf.get('confidence', 0)

                        mn1_trend = 'N/A'
                        w1_trend  = 'N/A'

                        if raw_mn1 and len(raw_mn1) >= 10:
                            mn1_df    = self._candles_to_df(raw_mn1)
                            mn1_htf   = self.smc.determine_htf_trend(mn1_df)
                            mn1_trend = mn1_htf.get('trend', 'UNKNOWN')

                        if raw_w1 and len(raw_w1) >= 10:
                            w1_df    = self._candles_to_df(raw_w1)
                            w1_htf   = self.smc.determine_htf_trend(w1_df)
                            w1_trend = w1_htf.get('trend', 'UNKNOWN')

                        # Determine alignment quality
                        trends_known = [
                            t for t in [mn1_trend, w1_trend, d1_trend]
                            if t not in ('UNKNOWN', 'N/A', 'RANGING')
                        ]
                        if len(trends_known) >= 2 and len(set(trends_known)) == 1:
                            alignment = 'ALIGNED'
                        elif len(trends_known) >= 2 and len(set(trends_known)) > 1:
                            alignment = 'MIXED'
                        else:
                            alignment = 'RANGING'

                        lines.append(
                            f"  {symbol:<10}  "
                            f"MN1: {mn1_trend:<8}  "
                            f"W1: {w1_trend:<8}  "
                            f"D1: {d1_trend:<8} ({d1_conf}%)  "
                            f"[{alignment}]"
                        )
                    except Exception as _sym_err:
                        self.logger.warning(
                            "Weekly analysis failed for %s: %s", symbol, _sym_err)
                        lines.append(f"  {symbol:<10} Analysis unavailable")

                lines += [
                    "",
                    "ALIGNMENT KEY:",
                    "  ALIGNED  - All timeframes agree. Highest probability setups.",
                    "  MIXED    - Timeframes disagree. Trade with reduced size or wait.",
                    "  RANGING  - No clear direction. Avoid until structure forms.",
                    "",
                ]
            else:
                lines.append("  Weekly bias unavailable. MT5 worker is offline.")

            # Upcoming week news
            try:
                events = self.news.get_red_folder_events(hours_ahead=168)
                if events:
                    news_lines = []
                    user_tz = user_now.tzinfo
                    for ev in events[:8]:
                        try:
                            import pytz
                            ev_utc = ev.timestamp
                            if ev_utc.tzinfo is None:
                                ev_utc = pytz.utc.localize(ev_utc)
                            ev_local  = ev_utc.astimezone(user_tz)
                            day_str   = ev_local.strftime('%a')
                            time_str  = ev_local.strftime('%I:%M %p').lstrip('0')
                            local_str = f"{day_str} {time_str} (your time)"
                        except Exception:
                            local_str = 'Unknown'
                        currency = getattr(ev, 'currency', 'N/A')
                        title    = getattr(ev, 'title', str(ev))
                        news_lines.append(f"  {local_str}  {currency:<4}  {title}")
                    news_body = "\n".join(news_lines)
                else:
                    news_body = "  Check Forex Factory for the week ahead."
            except Exception:
                news_body = "  News data unavailable."

            lines += [
                "",
                "KEY NEWS EVENTS THIS WEEK:",
                "",
                news_body,
                "",
                "TRADING GUIDELINES:",
                "",
                "  - Only trade in the direction of the weekly bias",
                "  - Avoid entries 30 min before and 15 min after red events",
                "  - Best sessions: London Open and New York Open are highest volume",
                "",
                config.FOOTER,
            ]

            message = utils.validate_user_message("\n".join(lines))
            await self._safe_send(telegram_id, message)

        except Exception as e:
            self.logger.error("Error in _send_weekly_analysis for user %d: %s", telegram_id, e)

    # ==================== SAFE SEND ====================

    async def _safe_send(self, telegram_id: int, text: str) -> bool:
        """
        Send a message. If user is unreachable, queue it for later delivery.
        Uses SYSTEM_MESSAGE type which is one of the three valid values in the
        message_queue CHECK constraint. SCHEDULED_ALERT is not a valid type
        and would cause a database constraint violation on every failed send.
        """
        try:
            await self.bot.send_message(chat_id=telegram_id, text=text)
            return True
        except Forbidden:
            self.logger.info("User %d blocked the bot. Message not sent.", telegram_id)
            return False
        except NetworkError as e:
            self.logger.warning(
                "Network error sending to user %d: %s. Queuing.", telegram_id, e)
            try:
                db.queue_message(telegram_id, text, 'SYSTEM_MESSAGE')
            except Exception as queue_err:
                self.logger.error(
                    "Failed to queue message for user %d: %s", telegram_id, queue_err)
            return False
        except Exception as e:
            self.logger.warning("Send error for user %d: %s", telegram_id, e)
            try:
                db.queue_message(telegram_id, text, 'SYSTEM_MESSAGE')
            except Exception:
                pass
            return False

    # ==================== TEST TRIGGER METHODS ====================

    async def trigger_daily_briefing_now(self, telegram_id: int):
        """Force-send the 06:30 daily briefing immediately. Admin testing only."""
        import pytz
        user = db.get_user(telegram_id)
        tz_str = (
            user.get('timezone') or user.get('user_timezone') or 'UTC'
        ) if user else 'UTC'
        try:
            user_tz = pytz.timezone(tz_str)
        except Exception:
            user_tz = pytz.utc
        user_now = datetime.now(timezone.utc).astimezone(user_tz)
        await self._send_daily_briefing(telegram_id, user_now)

    async def trigger_news_alert_now(self, telegram_id: int):
        """Force-send the 08:00 news alert immediately. Admin testing only."""
        import pytz
        user = db.get_user(telegram_id)
        tz_str = (
            user.get('timezone') or user.get('user_timezone') or 'UTC'
        ) if user else 'UTC'
        try:
            user_tz = pytz.timezone(tz_str)
        except Exception:
            user_tz = pytz.utc
        user_now = datetime.now(timezone.utc).astimezone(user_tz)
        await self._send_news_alert(telegram_id, user_now)

    async def trigger_weekly_analysis_now(self, telegram_id: int):
        """Force-send the Sunday weekly analysis immediately. Admin testing only."""
        import pytz
        user = db.get_user(telegram_id)
        tz_str = (
            user.get('timezone') or user.get('user_timezone') or 'UTC'
        ) if user else 'UTC'
        try:
            user_tz = pytz.timezone(tz_str)
        except Exception:
            user_tz = pytz.utc
        user_now = datetime.now(timezone.utc).astimezone(user_tz)
        await self._send_weekly_analysis(telegram_id, user_now)

    async def trigger_market_scan_now(self):
        """Force-run a full market scan immediately. Admin testing only."""
        await self.scan_markets()
