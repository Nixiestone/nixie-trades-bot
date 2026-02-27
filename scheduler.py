import asyncio
import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
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
    Runs two recurring background tasks:
      - Daily 8 AM UTC market overview with news summary
      - Market scan every 15 minutes for new SMC setups
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
        self.logger.info("Scheduler initialised.")

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
            )
            self.scheduler.add_job(
                self.scan_markets,
                CronTrigger(minute='*/15', timezone='UTC'),
                id='market_scan',
                name='Market Scan Every 15 Minutes',
                replace_existing=True,
            )
            self.scheduler.start()
            self.running = True
            self.logger.info(
                "Scheduler started. Daily alert: 08:00 UTC. "
                "Market scan: every 15 minutes.")
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
                f"The system scans all pairs every 15 minutes. "
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

    # ==================== MARKET SCAN ====================

    async def scan_markets(self):
        """Scan all monitored symbols every 15 minutes for qualifying setups."""
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
            # Check news blackout window before doing any analysis
            is_blackout, news_event = self.news.check_news_blackout(symbol)
            if is_blackout:
                self.logger.debug(
                    "Skipping %s: news blackout for %s %s.",
                    symbol,
                    getattr(news_event, 'currency', ''),
                    getattr(news_event, 'title', ''))
                return

            # Avoid duplicate setups within the last 15 minutes
            if db.recent_signal_exists(symbol, minutes=15):
                self.logger.debug(
                    "Skipping %s: setup already generated in last 15 minutes.", symbol)
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

            def to_df(raw):
                df = pd.DataFrame(raw)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)
                return df

            d1_df  = to_df(d1_raw)
            h4_df  = to_df(h4_raw)  if h4_raw  else None
            h1_df  = to_df(h1_raw)
            m15_df = to_df(m15_raw)
            m5_df  = to_df(m5_raw)  if m5_raw  else None

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
                        for m5_ob in m5_obs[:3]:
                            m5_entry_candidate = (
                                m5_ob.get('low', entry_cfg['entry_price'])
                                if poi['direction'] == 'BULLISH'
                                else m5_ob.get('high', entry_cfg['entry_price'])
                            )
                            if poi_low <= m5_entry_candidate <= poi_high:
                                self.logger.info(
                                    "M5 refinement for %s: entry %s -> %s",
                                    symbol,
                                    entry_cfg['entry_price'],
                                    round(m5_entry_candidate, 5))
                                entry_cfg['entry_price'] = round(m5_entry_candidate, 5)
                                break
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
                mins = int((next_news.timestamp - datetime.utcnow()).total_seconds() / 60)
                if 0 < mins <= 240:
                    news_warn = (
                        f"Note: {next_news.currency} {next_news.title} "
                        f"scheduled in {mins} minutes. Exercise caution.")

            setup_data = {
                'symbol':        symbol,
                'direction':     'BUY' if poi['direction'] == 'BULLISH' else 'SELL',
                'setup_type':    f"{'Unicorn' if tier == 'PREMIUM' else 'Standard'} {setup_type}",
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
        """Run ATR volatility and session filters before broadcasting."""
        try:
            atr = self.smc._calculate_atr(data.tail(20))
            atr_avg = float(data.tail(40)['close'].diff().abs().mean())
            pass_atr, reason = self.smc.check_atr_filter(atr, atr_avg)
            if not pass_atr:
                self.logger.debug("%s failed ATR filter: %s", symbol, reason)
                return False

            pass_session, reason = self.smc.check_session_filter(datetime.utcnow())
            if not pass_session:
                self.logger.debug("%s failed session filter: %s", symbol, reason)
                return False

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
            daily_loss = db.get_daily_loss_percent(tid)
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
                magic_number=config.MAGIC_NUMBER,
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
                    mt5_ticket=ticket,
                    symbol=symbol,
                    direction=direction,
                    lot_size=lot_size,
                    entry_price=entry,
                    stop_loss=setup_data['stop_loss'],
                    take_profit_1=setup_data['take_profit_1'],
                    take_profit_2=setup_data['take_profit_2'],
                    order_type=order_type,
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
                    df = pd.DataFrame(raw)
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                    df.sort_index(inplace=True)

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