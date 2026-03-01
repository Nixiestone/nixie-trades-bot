import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
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
                self.daily_alert,
                CronTrigger(hour=8, minute=0, timezone='UTC'),
                id='daily_alert',
                name='Daily 8 AM Market Overview',
                replace_existing=True,
                misfire_grace_time=3600,
                coalesce=True,
                max_instances=1,
            )
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
                f"{datetime.utcnow().strftime('%A, %B %d, %Y')} (UTC)\n\n"
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

    # ==================== MARKET SCAN ====================

    async def scan_markets(self):
        """Scan all monitored symbols on the configured scheduler interval."""
        try:
            self.logger.info("Starting market scan.")
            if not self.mt5.is_worker_reachable():
                self.logger.warning(
                    "MT5 worker not reachable. Skipping market scan. "
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
            if not self.mt5.is_worker_reachable():
                self.logger.debug(
                    "MT5 worker not reachable. Skipping %s.", symbol)
                return

            # Check news blackout window before doing any analysis
            is_blackout, news_event = self.news.check_news_blackout(symbol)
            if is_blackout:
                self.logger.debug(
                    "Skipping %s: news blackout for %s %s.",
                    symbol,
                    getattr(news_event, 'currency', ''),
                    getattr(news_event, 'title', ''))
                return

            # Avoid duplicate setups within one market-scan interval.
            if db.recent_signal_exists(symbol, minutes=self.market_scan_interval_minutes):
                self.logger.debug(
                    "Skipping %s: setup already generated in the last %d minutes.",
                    symbol,
                    self.market_scan_interval_minutes,
                )
                return

            # Fetch multi-timeframe data
            # D1 provides HTF context, H4 provides intermediate confirmation,
            # H1 provides structure, M15 provides entry timing, M5 provides
            # precise entry refinement within the H1 POI zone.
            d1_raw  = self.mt5.get_historical_data(symbol, 'D1',  bars=250)
            h4_raw  = self.mt5.get_historical_data(symbol, 'H4',  bars=300)
            h1_raw  = self.mt5.get_historical_data(symbol, 'H1',  bars=500)
            m15_raw = self.mt5.get_historical_data(symbol, 'M15', bars=500)
            m5_raw  = self.mt5.get_historical_data(symbol, 'M5',  bars=300)

            if not d1_raw or len(d1_raw) < 50:
                self.logger.debug("Insufficient D1 data for %s. Skipping.", symbol)
                return
            if not h1_raw or len(h1_raw) < 100:
                self.logger.debug("Insufficient H1 data for %s. Skipping.", symbol)
                return
            if not m15_raw or len(m15_raw) < 100:
                self.logger.debug("Insufficient M15 data for %s. Skipping.", symbol)
                return

            d1_df  = self._candles_to_df(d1_raw)
            h4_df  = self._candles_to_df(h4_raw) if h4_raw else None
            h1_df  = self._candles_to_df(h1_raw)
            m15_df = self._candles_to_df(m15_raw)
            m5_df  = self._candles_to_df(m5_raw) if m5_raw else None

            # Phase 1: HTF trend context (D1)
            htf_trend = self.smc.determine_htf_trend(d1_df)
            if htf_trend.get('trend') == 'RANGING':
                self.logger.debug("%s D1 trend is RANGING. Skipping.", symbol)
                return

            # H4 intermediate confirmation - must agree with D1 direction.
            # If H4 is ranging or opposing D1, skip to avoid counter-trend entries.
            h4_aligned = False
            if h4_df is not None and len(h4_df) >= 50:
                h4_trend = self.smc.determine_htf_trend(h4_df)
                h4_aligned = h4_trend.get('trend') == htf_trend.get('trend')
                if not h4_aligned and h4_trend.get('trend') != 'RANGING':
                    self.logger.debug(
                        "%s H4 trend (%s) opposes D1 trend (%s). Skipping.",
                        symbol, h4_trend.get('trend'), htf_trend.get('trend'))
                    return
                
            # Phase 2: Intermediate structure (H1)
            bos_events = self.smc.detect_break_of_structure(h1_df, htf_trend['trend'])
            mss_event  = self.smc.detect_market_structure_shift(h1_df, htf_trend['trend'])

            setup_type = None
            poi        = None

            if len(bos_events) >= 2:
                setup_type = 'BOS'
                breakers   = self.smc.detect_breaker_blocks(
                    h1_df, htf_trend['trend'],
                    htf_trend.get('swing_high', 0),
                    htf_trend.get('swing_low', 0))
                if breakers:
                    poi = breakers[0]
            elif mss_event:
                setup_type = 'MSS'
                obs = self.smc.detect_order_blocks(h1_df, mss_event['direction'])
                if obs:
                    poi = obs[0]

            if poi is None or setup_type is None:
                self.logger.debug(
                    "%s: no valid structure or POI found.", symbol)
                return

            # Phase 3: ML validation
            ml_result     = self.ml.get_ensemble_prediction(
                m15_df, poi, htf_trend, setup_type)
            consensus     = ml_result['consensus_score']
            should_send, tier = self.ml.should_send_setup(consensus)

            if not should_send:
                self.logger.debug(
                    "%s: ML consensus %d%% below threshold. Rejected.", symbol, consensus)
                return

            # ATR and session filters
            if not self._check_filters(symbol, m15_df, htf_trend, poi):
                return

            # Phase 4: Entry / SL / TP calculation
            entry_cfg = self.smc.calculate_entry_price(
                poi, 'UNICORN' if tier == 'PREMIUM' else 'STANDARD', consensus)
            sl_cfg    = self.smc.calculate_stop_loss(
                poi, poi['direction'], symbol,
                self.smc._calculate_atr(m15_df.tail(20)))
            
            # M5 precision entry refinement.
            # If there is an M5 Order Block inside the H1 POI zone,
            # tighten the entry to that M5 level for a better R:R.
            if m5_df is not None and len(m5_df) >= 50:
                try:
                    m5_obs = self.smc.detect_order_blocks(
                        m5_df.tail(50), poi['direction'])
                    if m5_obs:
                        poi_high = float(poi.get('high', entry_cfg['entry_price']))
                        poi_low  = float(poi.get('low',  entry_cfg['entry_price']))
                        candidates = []
                        for m5_ob in m5_obs[:3]:
                            m5_entry_candidate = (
                                m5_ob.get('low', entry_cfg['entry_price'])
                                if poi['direction'] == 'BULLISH'
                                else m5_ob.get('high', entry_cfg['entry_price'])
                            )
                            if poi_low <= m5_entry_candidate <= poi_high:
                                candidates.append(float(m5_entry_candidate))
                        if candidates:
                            best_candidate = (
                                min(candidates)
                                if poi['direction'] == 'BULLISH'
                                else max(candidates)
                            )
                            self.logger.info(
                                "M5 refinement for %s: entry %s -> %s",
                                symbol,
                                entry_cfg['entry_price'],
                                round(best_candidate, 5))
                            entry_cfg['entry_price'] = round(best_candidate, 5)
                except Exception as m5_err:
                    self.logger.debug(
                        "M5 refinement skipped for %s: %s", symbol, m5_err)
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
                mins = int((event_ts - datetime.now(timezone.utc)).total_seconds() / 60)
                if 0 < mins <= 240:
                    news_warn = (
                        f"Note: {next_news.currency} {next_news.title} "
                        f"scheduled in {mins} minutes. Exercise caution.")

            setup_tier  = 'UNICORN' if tier == 'PREMIUM' else 'STANDARD'
            setup_label = f"{setup_tier} {setup_type}"

            setup_data = {
                'symbol':        symbol,
                'direction':     'BUY' if poi['direction'] == 'BULLISH' else 'SELL',
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
            }

            signal_id = db.save_signal(
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
                order_type='LIMIT',
            )

            if signal_id:
                setup_data['signal_number'] = signal_id
                auto_execute = self.ml.should_auto_execute(consensus)
                await self._broadcast_setup(setup_data, auto_execute)
                self.logger.info(
                    "Setup generated and broadcast: %s %s | Score: %d%% | "
                    "Tier: %s | Signal ID: %s",
                    symbol, setup_data['direction'], consensus, tier, signal_id)

        except Exception as e:
            self.logger.error("Error scanning %s: %s", symbol, e, exc_info=True)

    def _check_filters(
        self, symbol: str, data: pd.DataFrame, htf_trend: dict, poi: dict
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

            # Filter 1: ATR volatility
            atr     = self.smc._calculate_atr(data.tail(20))
            atr_avg = float(data.tail(40)['close'].diff().abs().mean())
            pass_atr, reason = self.smc.check_atr_filter(atr, atr_avg)
            if not pass_atr:
                self.logger.debug("%s failed ATR filter: %s", symbol, reason)
                return False

            # Filter 2: Session
            pass_session, reason = self.smc.check_session_filter(datetime.utcnow())
            if not pass_session:
                self.logger.debug("%s failed session filter: %s", symbol, reason)
                return False

            # Filter 3: ADX - reject ranging markets at entry timeframe
            adx_data = data.tail(60)
            if len(adx_data) >= 33:
                pass_adx, adx_val, reason_adx = self.smc.check_adx_filter(adx_data)
                if not pass_adx:
                    self.logger.debug(
                        "%s failed ADX filter: %s (ADX=%.1f)", symbol, reason_adx, adx_val)
                    return False

            # Filter 4: Premium/discount zone
            swing_high    = float(htf_trend.get('swing_high', 0))
            swing_low     = float(htf_trend.get('swing_low',  0))
            current_price = float(data.iloc[-1]['close'])
            if swing_high > swing_low > 0:
                in_zone, zone_pos, zone_desc = self.smc.detect_premium_discount_zone(
                    current_price, swing_high, swing_low, direction)
                if not in_zone:
                    self.logger.debug(
                        "%s failed zone filter: %s", symbol, zone_desc)
                    return False
                if direction == 'BULLISH' and zone_pos > 0.45:
                    self.logger.debug(
                        "%s failed sniper depth filter: bullish zone %.2f > 0.45",
                        symbol, zone_pos)
                    return False
                if direction == 'BEARISH' and zone_pos < 0.55:
                    self.logger.debug(
                        "%s failed sniper depth filter: bearish zone %.2f < 0.55",
                        symbol, zone_pos)
                    return False

            # Filter 5: Composite setup quality score
            try:
                bos_events = self.smc.detect_break_of_structure(
                    data.tail(50), htf_trend.get('trend', 'BULLISH'))
                quality = self.smc.score_setup_quality(
                    data, poi, htf_trend, bos_events, direction, symbol)
                if quality < 60:
                    self.logger.debug(
                        "%s failed quality filter: score %d < 60", symbol, quality)
                    return False
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
            bid, ask = await asyncio.get_event_loop().run_in_executor(
                None, self.mt5.get_current_price, symbol
            )
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

            sent = 0
            for user in subscribed:
                tid = user['telegram_id']
                try:
                    # Calculate lot size per user's risk setting if MT5 connected
                    lot_size = None
                    if user.get('mt5_connected') and auto_execute:
                        lot_size = await self._calculate_user_lot_size(user, setup_data)

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
                    )

                    if news_warn:
                        message += f"\n\n{news_warn}"

                    try:
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
                    if auto_execute and user.get('mt5_connected') and lot_size:
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

            ok, account_info = self.mt5.get_account_info(user['telegram_id'])
            if not ok or not account_info:
                return None

            balance  = float(account_info.get('balance', 0))
            currency = account_info.get('currency', 'USD')
            risk_pct = float(user.get('risk_percent', config.DEFAULT_RISK_PERCENT))
            sl_pips  = float(setup_data.get('sl_pips', 20.0))

            if balance <= 0 or sl_pips <= 0:
                return None

            exchange_rates = self.mt5.get_exchange_rates()
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
            bid, ask = self.mt5.get_current_price(symbol)
            current  = ask if direction == 'BUY' else bid
            entry    = setup_data['entry_price']

            # If price is already at or past the entry, use MARKET
            if direction == 'BUY':
                order_type = 'MARKET' if (current is not None and current <= entry) else 'LIMIT'
            else:
                order_type = 'MARKET' if (current is not None and current >= entry) else 'LIMIT'

            success, ticket, message = self.mt5.place_order(
                telegram_id=tid,
                symbol=symbol,
                direction=direction,
                lot_size=lot_size,
                entry_price=entry,
                stop_loss=setup_data['stop_loss'],
                take_profit=setup_data['take_profit_1'],   # Initial TP is TP1
                order_type=order_type,
                comment='Nix Trades Auto',
            )

            if success and ticket:
                self.logger.info(
                    "Auto-executed trade for user %d: %s %s %.2f lots "
                    "at %.5f (ticket %d).",
                    tid, direction, symbol, lot_size, entry, ticket)

                # Register with position monitor for TP1/TP2 automation
                if self.monitor is not None:
                    self.monitor.add_position(
                        ticket=ticket,
                        symbol=symbol,
                        direction=direction,
                        volume=lot_size,
                        entry_price=entry,
                        stop_loss=setup_data['stop_loss'],
                        take_profit_1=setup_data['take_profit_1'],
                        take_profit_2=setup_data['take_profit_2'],
                        telegram_id=tid,
                        magic_number=config.MAGIC_NUMBER,
                        ml_features=setup_data.get('ml_features'),
                    )

                # Save trade record to database
                db.save_trade(
                    telegram_id=tid,
                    signal_id=setup_data.get('signal_number'),
                    symbol=symbol,
                    direction=direction,
                    lot_size=lot_size,
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
            if not self.mt5.is_worker_reachable():
                return "Market data unavailable (MT5 worker not connected)."

            lines = ["MARKET STRUCTURE:"]
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']

            for symbol in major_pairs:
                try:
                    raw = self.mt5.get_historical_data(symbol, 'D1', bars=50)
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
        """
        try:
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
                mt5_ok = self.mt5.is_worker_reachable()
            except Exception:
                mt5_ok = False

            if mt5_ok:
                for symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'GBPJPY', 'AUDUSD']:
                    try:
                        raw = self.mt5.get_historical_data(symbol, 'D1', bars=100)
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
        """
        try:
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
        """
        try:
            date_str = user_now.strftime('%A, %B %d, %Y')

            lines = [
                "WEEKLY MARKET ANALYSIS",
                date_str,
                "",
                "Good morning. Here is your weekly outlook for the week ahead.",
                "",
            ]

            try:
                mt5_ok = self.mt5.is_worker_reachable()
            except Exception:
                mt5_ok = False

            if mt5_ok:
                lines += ["WEEKLY BIAS (Daily Structure):", ""]
                for symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
                               'USDCAD', 'NZDUSD', 'XAUUSD', 'GBPJPY', 'EURJPY']:
                    try:
                        raw_d1 = self.mt5.get_historical_data(symbol, 'D1', bars=50)
                        if not raw_d1:
                            continue
                        df = self._candles_to_df(raw_d1)
                        htf        = self.smc.determine_htf_trend(df)
                        trend      = htf.get('trend', 'UNKNOWN')
                        confidence = htf.get('confidence', 0)
                        lines.append(f"  {symbol:<10} {trend:<8}  ({confidence}% confidence)")
                    except Exception:
                        lines.append(f"  {symbol:<10} Data unavailable")
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
        """
        try:
            await self.bot.send_message(chat_id=telegram_id, text=text)
            return True
        except Forbidden:
            self.logger.info("User %d blocked the bot. Message not sent.", telegram_id)
            return False
        except NetworkError as e:
            self.logger.warning("Network error sending to user %d: %s. Queuing.", telegram_id, e)
            try:
                db.queue_message(telegram_id, text, 'SCHEDULED_ALERT')
            except Exception:
                pass
            return False
        except Exception as e:
            self.logger.warning("Send error for user %d: %s", telegram_id, e)
            return False

    # ==================== TEST TRIGGER METHODS ====================

    async def trigger_daily_briefing_now(self, telegram_id: int):
        """Force-send the 06:30 daily briefing immediately. Admin testing only."""
        await self._send_daily_briefing(telegram_id, datetime.now(timezone.utc))

    async def trigger_news_alert_now(self, telegram_id: int):
        """Force-send the 08:00 news alert immediately. Admin testing only."""
        await self._send_news_alert(telegram_id, datetime.now(timezone.utc))

    async def trigger_weekly_analysis_now(self, telegram_id: int):
        """Force-send the Sunday weekly analysis immediately. Admin testing only."""
        await self._send_weekly_analysis(telegram_id, datetime.now(timezone.utc))

    async def trigger_market_scan_now(self):
        """Force-run a full market scan immediately. Admin testing only."""
        await self.scan_markets()
