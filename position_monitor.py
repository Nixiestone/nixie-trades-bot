import asyncio
import logging
import threading
import time as time_module
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import config
import database as db
import utils

logger = logging.getLogger(__name__)


@dataclass
class MonitoredPosition:
    """One live trade being watched by the position monitor."""
    ticket:        int
    symbol:        str
    direction:     str          # 'BUY' or 'SELL'
    volume:        float        # Lot size
    entry_price:   float
    current_price: float
    stop_loss:     float
    take_profit_1: float
    take_profit_2: float
    status:        str          # 'ACTIVE', 'TP1_HIT', 'TP2_HIT', 'STOPPED'
    telegram_id:   int
    magic_number:  int          = 234567
    opened_at:     datetime     = field(default_factory=datetime.now)
    last_check:    datetime     = field(default_factory=datetime.now)
    tp1_closed:    bool         = False
    be_activated:  bool         = False
    profit:        float        = 0.0
    ml_features:              Optional[object] = None
    awaiting_partial_confirm: bool  = False
    partial_confirm_sent_at:  float = 0.0


class PositionMonitor:
    """
    Background thread that watches all open trades every 10 seconds.
    Handles TP1 partial close, breakeven, TP2 full close, and stop loss.
    Sends Telegram notifications for every event.
    """

    def __init__(self, mt5_connector, database=None, telegram_bot=None):
        self.logger               = logging.getLogger(f"{__name__}.PositionMonitor")
        self.mt5                  = mt5_connector
        self.db                   = database       # database module
        self.bot                  = telegram_bot   # telegram.Bot instance
        self.monitored_positions: Dict[int, MonitoredPosition] = {}
        self.running              = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.check_interval       = config.POSITION_CHECK_INTERVAL_SECONDS
        # The asyncio event loop from the main thread - captured at start()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.logger.info("Position Monitor initialised.")

    # ==================== LIFECYCLE ====================

    def start(self) -> bool:
        """
        Start the position monitor in a daemon background thread.
        Captures the running event loop so sends can be scheduled from the thread.
        Call this from bot._post_init() which runs on the event loop.
        """
        if self.running:
            self.logger.warning("Position monitor is already running.")
            return True
        try:
            # Capture the event loop currently running in the main thread.
            # This works because start() is called from _post_init which runs
            # on the bot's event loop.
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    self._loop = asyncio.get_event_loop()
                except Exception:
                    self._loop = None

            self.running        = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name='PositionMonitorThread',
            )
            self.monitor_thread.start()
            self.logger.info("Position monitor background thread started.")
            return True
        except Exception as e:
            self.logger.error("Failed to start position monitor: %s", e)
            self.running = False
            return False

    def stop(self) -> bool:
        """Signal the background thread to stop and wait up to 30 seconds."""
        if not self.running:
            return True
        self.logger.info("Stopping position monitor...")
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=30)
            if self.monitor_thread.is_alive():
                self.logger.warning(
                    "Position monitor thread did not stop within 30 seconds."
                )
        self.logger.info("Position monitor stopped.")
        return True

    # ==================== MAIN LOOP ====================

    def _monitor_loop(self):
        """Background thread: check all positions at each interval."""
        self.logger.info(
            "Position monitor loop started. Checking every %d seconds.",
            self.check_interval,
        )
        while self.running:
            try:
                self._check_all_positions()
            except Exception as e:
                self.logger.error(
                    "Unexpected error in position monitor loop: %s", e, exc_info=True
                )
            time_module.sleep(self.check_interval)
        self.logger.info("Position monitor loop ended.")

    # ==================== POSITION MANAGEMENT ====================

    def add_position(
        self,
        ticket:        int,
        symbol:        str,
        direction:     str,
        volume:        float,
        entry_price:   float,
        stop_loss:     float,
        take_profit_1: float,
        take_profit_2: float,
        telegram_id:   int,
        magic_number:  int   = 234567,
        ml_features=None,
    ) -> bool:
        """Register a newly opened trade for monitoring."""
        try:
            position = MonitoredPosition(
                ticket=ticket,
                symbol=symbol,
                direction=direction,
                volume=volume,
                entry_price=entry_price,
                current_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                status='ACTIVE',
                telegram_id=telegram_id,
                magic_number=magic_number,
                opened_at=datetime.now(),
                ml_features=ml_features,
            )
            self.monitored_positions[ticket] = position
            self.logger.info(
                "Position added to monitor: ticket=%d %s %s %.2f lots at %.5f.",
                ticket, direction, symbol, volume, entry_price,
            )
            self._send_notification(
                telegram_id,
                f"POSITION OPENED\n\n"
                f"Ticket:    {ticket}\n"
                f"Symbol:    {symbol}\n"
                f"Direction: {direction}\n"
                f"Volume:    {volume:.2f} lots\n"
                f"Entry:     {utils.format_price(symbol, entry_price)}\n"
                f"Stop Loss: {utils.format_price(symbol, stop_loss)}\n"
                f"TP1:       {utils.format_price(symbol, take_profit_1)}\n"
                f"TP2:       {utils.format_price(symbol, take_profit_2)}\n\n"
                f"Position is now being monitored automatically."
            )
            return True
        except Exception as e:
            self.logger.error("Error adding position %d to monitor: %s", ticket, e)
            return False

    def remove_position(self, ticket: int) -> bool:
        """Remove a position from monitoring after it closes."""
        if ticket in self.monitored_positions:
            del self.monitored_positions[ticket]
            self.logger.info("Position %d removed from monitor.", ticket)
            return True
        return False

    def get_all_positions(self) -> List[MonitoredPosition]:
        """Return all currently monitored positions."""
        return list(self.monitored_positions.values())
    
    def get_position_by_ticket(self, ticket: int) -> Optional[MonitoredPosition]:
        """Return the MonitoredPosition for a given MT5 ticket, or None."""
        return self.monitored_positions.get(ticket)

    # ==================== CHECKING LOGIC ====================

    def _check_all_positions(self):
        """Check every monitored position against live MT5 data."""
        if not self.monitored_positions:
            return

        try:
            mt5_positions = self.mt5.get_open_positions()
        except Exception as e:
            self.logger.error("Could not get open positions from MT5: %s", e)
            return

        mt5_map          = {int(p.get('ticket', 0)): p for p in (mt5_positions or [])}
        tickets_to_remove = []

        for ticket, position in list(self.monitored_positions.items()):
            try:
                if ticket not in mt5_map:
                    # No longer in MT5 - stop loss hit or manually closed
                    self._handle_position_closed(position)
                    tickets_to_remove.append(ticket)
                    continue

                live                   = mt5_map[ticket]
                position.current_price = float(live.get('price_current', position.current_price))
                position.profit        = float(live.get('profit', 0.0))
                position.last_check    = datetime.now()
                
                # Expire partial profit confirmations after 5 minutes.
                # If user did not respond, hold full position to TP2.
                if (position.awaiting_partial_confirm
                        and position.partial_confirm_sent_at > 0
                        and time_module.time() - position.partial_confirm_sent_at > 300):
                    self.logger.info(
                        "Ticket %d: partial profit confirmation expired. "
                        "Holding full position to TP2.", position.ticket)
                    position.awaiting_partial_confirm = False

                if not position.tp1_closed and self._check_tp1_hit(position):
                    self._handle_tp1_hit(position)

                if position.tp1_closed and not position.be_activated:
                    self._handle_be_activation(position)

                if position.tp1_closed and self._check_tp2_hit(position):
                    self._handle_tp2_hit(position)
                    tickets_to_remove.append(ticket)

            except Exception as e:
                self.logger.error(
                    "Error processing position %d: %s", ticket, e, exc_info=True
                )

        for ticket in tickets_to_remove:
            self.remove_position(ticket)

    # ==================== PRICE CHECKS ====================

    def _check_tp1_hit(self, position: MonitoredPosition) -> bool:
        try:
            if position.direction == 'BUY':
                return position.current_price >= position.take_profit_1
            return position.current_price <= position.take_profit_1
        except Exception as e:
            self.logger.error("Error checking TP1 for ticket %d: %s", position.ticket, e)
            return False

    def _check_tp2_hit(self, position: MonitoredPosition) -> bool:
        try:
            if position.direction == 'BUY':
                return position.current_price >= position.take_profit_2
            return position.current_price <= position.take_profit_2
        except Exception as e:
            self.logger.error("Error checking TP2 for ticket %d: %s", position.ticket, e)
            return False

    # ==================== EVENT HANDLERS ====================

    def _handle_tp1_hit(self, position: MonitoredPosition):
        """
        TP1 has been reached.

        For positions with minimum lot size (0.01 lots): partial close is
        mathematically impossible because half of 0.01 = 0.005, which is
        below every broker's minimum volume. Instead, move SL to breakeven
        and hold the full position to TP2 to protect the trade.

        For all other positions: send a Telegram confirmation asking the
        user whether to close 50% now. Do NOT close automatically.
        The trade stays open until the user responds or 5 minutes elapse.
        """
        self.logger.info("TP1 hit for ticket %d (%s).", position.ticket, position.symbol)

        if position.tp1_closed or position.awaiting_partial_confirm:
            return

        position.tp1_closed = True
        position.status     = 'TP1_HIT'

        # Minimum lot for a valid 50% partial close.
        # Half of 0.01 = 0.005, which brokers reject.
        MIN_LOT_FOR_PARTIAL = 0.02

        if position.volume < MIN_LOT_FOR_PARTIAL:
            # Cannot split - move SL to breakeven and run to TP2
            self.logger.info(
                "Ticket %d: lot size %.2f cannot be partially closed "
                "(minimum splittable lot is %.2f). "
                "Moving SL to breakeven and holding to TP2.",
                position.ticket, position.volume, MIN_LOT_FOR_PARTIAL)
            self._handle_be_activation(position)
            pips = utils.calculate_pips(
                position.symbol, position.entry_price, position.take_profit_1)
            self._send_notification(
                position.telegram_id,
                "TP1 REACHED - HOLDING TO TP2\n\n"
                "Ticket:    %d\n"
                "Symbol:    %s\n"
                "TP1 Price: %s\n"
                "Pips:      +%.1f\n\n"
                "Your position size (%.2f lots) cannot be partially closed "
                "as this would create a sub-minimum position.\n"
                "Stop loss has been moved to breakeven.\n"
                "Full position is running to TP2." % (
                    position.ticket,
                    position.symbol,
                    utils.format_price(position.symbol, position.take_profit_1),
                    pips,
                    position.volume,
                )
            )
        else:
            # Ask user for confirmation before closing 50%
            position.awaiting_partial_confirm = True
            position.partial_confirm_sent_at  = time_module.time()
            self._request_partial_profit_confirmation(position)

    def _request_partial_profit_confirmation(self, position: MonitoredPosition):
        """
        Send a Telegram inline keyboard asking the user whether to take
        50% partial profit at TP1. The trade is NOT closed until the user
        responds. If no response arrives within 5 minutes, the full
        position continues to TP2 (see timeout check in _check_all_positions).
        """
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup

            pips      = utils.calculate_pips(
                position.symbol, position.entry_price, position.take_profit_1)
            half_lots = round(position.volume / 2, 2)

            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton(
                    "Take 50 Percent Profit Now",
                    callback_data="partial_close_yes_%d" % position.ticket
                ),
                InlineKeyboardButton(
                    "Hold Full Position to TP2",
                    callback_data="partial_close_no_%d" % position.ticket
                ),
            ]])

            message = (
                "TP1 REACHED - YOUR DECISION REQUIRED\n\n"
                "Symbol:     %s\n"
                "Direction:  %s\n"
                "Ticket:     %d\n"
                "Pips:       +%.1f\n\n"
                "Take Profit 1 has been reached.\n\n"
                "Option 1: Close 50 percent (%.2f lots) now and let the "
                "remaining 50 percent run to TP2 with stop loss at breakeven.\n\n"
                "Option 2: Hold the full position open to TP2.\n\n"
                "If you do not respond within 5 minutes, the full position "
                "will continue to TP2 automatically.\n\n"
                "EDUCATIONAL PURPOSES ONLY. NOT FINANCIAL ADVICE."
            ) % (
                position.symbol,
                position.direction,
                position.ticket,
                pips,
                half_lots,
            )

            if self.bot is not None and self._loop is not None and self._loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self.bot.send_message(
                        chat_id=position.telegram_id,
                        text=message,
                        reply_markup=keyboard,
                    ),
                    self._loop,
                )
                try:
                    future.result(timeout=15)
                    self.logger.info(
                        "Partial profit confirmation sent to user %d for ticket %d.",
                        position.telegram_id, position.ticket)
                except Exception as send_err:
                    self.logger.warning(
                        "Failed to send partial confirmation for ticket %d: %s. "
                        "Defaulting to hold position.", position.ticket, send_err)
                    position.awaiting_partial_confirm = False
            else:
                # No event loop available - cannot show buttons, hold to TP2
                self.logger.warning(
                    "No event loop for partial confirmation ticket %d. "
                    "Holding full position to TP2.", position.ticket)
                position.awaiting_partial_confirm = False
                self._handle_be_activation(position)

        except Exception as e:
            self.logger.error(
                "Error sending partial confirmation for ticket %d: %s",
                position.ticket, e)
            position.awaiting_partial_confirm = False

    def _handle_tp2_hit(self, position: MonitoredPosition):
        """Close remaining 50% and record WIN."""
        self.logger.info("TP2 hit for ticket %d (%s).", position.ticket, position.symbol)
        try:
            success, message = self.mt5.close_position(
                position.ticket, percentage=100.0, comment='TP2 Full Close'
            )
            if success:
                position.status = 'TP2_HIT'
                pips            = utils.calculate_pips(
                    position.symbol, position.entry_price, position.take_profit_2
                )
                duration = self._trade_duration(position)
                self._send_notification(
                    position.telegram_id,
                    f"TP2 HIT - TRADE COMPLETE\n\n"
                    f"Ticket:   {position.ticket}\n"
                    f"Symbol:   {position.symbol}\n"
                    f"Entry:    {utils.format_price(position.symbol, position.entry_price)}\n"
                    f"TP2:      {utils.format_price(position.symbol, position.take_profit_2)}\n"
                    f"Pips:     +{pips:.1f}\n"
                    f"Duration: {duration}\n\n"
                    f"Full position closed. Trade complete."
                )
                self._log_trade_completion(position, 'WIN', position.profit)
                self.logger.info(
                    "TP2 processed for ticket %d: +%.1f pips, trade logged as WIN.",
                    position.ticket, pips,
                )
            else:
                self.logger.error(
                    "TP2 full close failed for ticket %d: %s", position.ticket, message
                )
        except Exception as e:
            self.logger.error("Error handling TP2 for ticket %d: %s", position.ticket, e)

    def _handle_be_activation(self, position: MonitoredPosition):
        """Move stop loss to entry price + buffer pips."""
        if position.be_activated:
            return
        try:
            pip_value = utils.get_pip_value(position.symbol)
            buffer    = config.BREAKEVEN_BUFFER_PIPS * pip_value
            be_price  = (
                position.entry_price + buffer if position.direction == 'BUY'
                else position.entry_price - buffer
            )
            success, message = self.mt5.modify_position(
                position.ticket,
                stop_loss=be_price,
                take_profit=position.take_profit_2,
            )
            if success:
                position.stop_loss    = be_price
                position.be_activated = True
                self.logger.info(
                    "Breakeven set for ticket %d at %.5f.", position.ticket, be_price
                )
            else:
                self.logger.error(
                    "Breakeven activation failed for ticket %d: %s",
                    position.ticket, message,
                )
        except Exception as e:
            self.logger.error(
                "Error activating breakeven for ticket %d: %s", position.ticket, e
            )

    def _handle_position_closed(self, position: MonitoredPosition):
        """Position disappeared from MT5 - stop loss or manual close."""
        self.logger.info(
            "Position %d (%s) is no longer in MT5. Recording as LOSS.",
            position.ticket, position.symbol,
        )
        try:
            position.status = 'STOPPED'
            duration        = self._trade_duration(position)
            self._send_notification(
                position.telegram_id,
                f"POSITION CLOSED\n\n"
                f"Ticket:    {position.ticket}\n"
                f"Symbol:    {position.symbol}\n"
                f"Direction: {position.direction}\n"
                f"Entry:     {utils.format_price(position.symbol, position.entry_price)}\n"
                f"Stop Loss: {utils.format_price(position.symbol, position.stop_loss)}\n"
                f"Duration:  {duration}\n\n"
                f"Stop loss closed the position to protect your account."
            )
            self._log_trade_completion(position, 'LOSS', position.profit)
        except Exception as e:
            self.logger.error(
                "Error handling closed position %d: %s", position.ticket, e
            )

    # ==================== NOTIFICATIONS ====================

    def _send_notification(self, telegram_id: int, message: str):
        """
        Send a message to the user from the background thread.

        How this works:
          The position monitor runs in a regular Python thread.
          Telegram's bot.send_message is an async function that must run
          on the asyncio event loop that the main bot uses.
          run_coroutine_threadsafe() is the official way to schedule an
          async function from a regular thread onto a running event loop.

          If no loop is available, the message is queued in the database
          and delivered the next time the user sends any command.
        """
        try:
            clean = utils.validate_user_message(f"{message}\n\n{config.FOOTER}")

            if self.bot is not None and self._loop is not None and self._loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self.bot.send_message(chat_id=telegram_id, text=clean),
                    self._loop,
                )
                try:
                    future.result(timeout=15)
                except Exception as send_err:
                    self.logger.warning(
                        "Telegram send failed for user %d (%s). Queuing message.",
                        telegram_id, send_err,
                    )
                    db.queue_message(telegram_id, clean, 'TRADE_NOTIFICATION')
            else:
                # Event loop not available - queue for next user interaction
                db.queue_message(telegram_id, clean, 'TRADE_NOTIFICATION')
                self.logger.debug(
                    "No event loop. Notification to user %d queued in database.",
                    telegram_id,
                )

        except Exception as e:
            self.logger.error(
                "Error sending notification to user %d: %s", telegram_id, e
            )

    # ==================== DATABASE LOGGING ====================

    def _log_trade_completion(
        self,
        position: MonitoredPosition,
        outcome:  str,
        profit:   float,
    ):
        """
        Write the completed trade result to the database.
        Also stores the ML feature vector and outcome so the ML model
        can learn from this trade when it auto-retrains after 100 outcomes.
        """
        try:
            pips = utils.calculate_pips(
                position.symbol, position.entry_price, position.current_price
            )
            if position.direction == 'SELL':
                pips = -pips

            sl_distance = abs(position.entry_price - position.stop_loss)
            rr = (
                abs(position.current_price - position.entry_price) / sl_distance
                if sl_distance > 0 else 0.0
            )

            # Update the trade record that was created when the order was placed
            db.update_trade(
                ticket=position.ticket,
                updates={
                    'status':       'CLOSED',
                    'close_price':  position.current_price,
                    'outcome':      outcome,
                    'profit_pips':  round(pips, 2),
                    'realized_pnl': round(profit, 2),
                    'rr_achieved':  round(rr, 2),
                    'closed_at':    datetime.now().isoformat(),
                    'updated_at':   datetime.now().isoformat(),
                },
            )
            self.logger.info(
                "Trade %d logged: outcome=%s pips=%.1f profit=%.2f rr=%.2f.",
                position.ticket, outcome, pips, profit, rr,
            )

            # Store ML outcome for auto-retraining
            if position.ml_features is not None:
                try:
                    db.save_trade_outcome_for_ml(
                        ticket=position.ticket,
                        features=position.ml_features,
                        outcome=1.0 if outcome == 'WIN' else 0.0,
                    )
                    self.logger.info(
                        "ML training data saved for ticket %d (outcome: %s).",
                        position.ticket, outcome,
                    )
                except Exception as ml_err:
                    self.logger.warning(
                        "Could not save ML training data for ticket %d: %s",
                        position.ticket, ml_err,
                    )

        except Exception as e:
            self.logger.error(
                "Error logging trade %d to database: %s", position.ticket, e, exc_info=True
            )

    # ==================== HELPERS ====================

    @staticmethod
    def _trade_duration(position: MonitoredPosition) -> str:
        """Return a human-readable trade duration string like '2h 15m'."""
        try:
            if position.opened_at is None:
                return 'unknown'
            delta   = datetime.now() - position.opened_at
            hours   = int(delta.total_seconds() // 3600)
            minutes = int((delta.total_seconds() % 3600) // 60)
            if hours > 0:
                return f"{hours}h {minutes}m"
            return f"{minutes}m"
        except Exception:
            return 'unknown'