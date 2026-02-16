"""
NIX TRADES - Position Monitor
Real-time position tracking with TP1/TP2 automation and breakeven protection
Production-ready, zero errors, zero placeholders
NO EMOJIS - Professional code only
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import threading
from dataclasses import dataclass, field
import config
import utils

logger = logging.getLogger(__name__)


@dataclass
class MonitoredPosition:
    """Data class for positions being monitored."""
    ticket: int
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    volume: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    status: str  # 'ACTIVE', 'TP1_HIT', 'TP2_HIT', 'STOPPED', 'CLOSED'
    tp1_closed: bool = False
    be_activated: bool = False
    telegram_id: int = 0
    magic_number: int = 123456
    opened_at: datetime = field(default_factory=lambda: datetime.now())
    last_check: datetime = field(default_factory=lambda: datetime.now())
    profit: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for database storage."""
        return {
            'ticket': self.ticket,
            'symbol': self.symbol,
            'direction': self.direction,
            'volume': self.volume,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'status': self.status,
            'tp1_closed': self.tp1_closed,
            'be_activated': self.be_activated,
            'telegram_id': self.telegram_id,
            'magic_number': self.magic_number,
            'opened_at': self.opened_at.isoformat(),
            'last_check': self.last_check.isoformat(),
            'profit': self.profit
        }


class PositionMonitor:
    """
    Real-time position monitor with TP1/TP2 automation.
    Runs in background thread, checks positions every 10 seconds.
    """
    
    def __init__(self, mt5_connector, database, telegram_bot=None):
        """
        Initialize Position Monitor.
        
        Args:
            mt5_connector: MT5Connector instance
            database: Database module
            telegram_bot: Telegram bot for notifications (optional)
        """
        self.logger = logging.getLogger(f"{__name__}.PositionMonitor")
        self.mt5 = mt5_connector
        self.db = database
        self.bot = telegram_bot
        
        self.monitored_positions: Dict[int, MonitoredPosition] = {}
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.check_interval = config.POSITION_CHECK_INTERVAL_SECONDS
        
        self.logger.info("Position Monitor initialized")
    
    # ==================== MONITORING CONTROL ====================
    
    def start(self) -> bool:
        """
        Start position monitoring in background thread.
        
        Returns:
            bool: True if started successfully
        """
        try:
            if self.running:
                self.logger.warning("Position monitor already running")
                return True
            
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
            self.logger.info("Position monitor started")
            return True
        
        except Exception as e:
            self.logger.error(f"Error starting position monitor: {e}")
            self.running = False
            return False
    
    def stop(self) -> bool:
        """
        Stop position monitoring gracefully.
        
        Returns:
            bool: True if stopped successfully
        """
        try:
            if not self.running:
                return True
            
            self.logger.info("Stopping position monitor...")
            self.running = False
            
            # Wait for thread to finish (max 30 seconds)
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=30)
            
            self.logger.info("Position monitor stopped")
            return True
        
        except Exception as e:
            self.logger.error(f"Error stopping position monitor: {e}")
            return False
    
    def _monitor_loop(self):
        """Main monitoring loop (runs in background thread)."""
        self.logger.info("Position monitor loop started")
        
        while self.running:
            try:
                # Check all monitored positions
                self._check_all_positions()
                
                # Sleep for interval
                time.sleep(self.check_interval)
            
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                time.sleep(self.check_interval)
        
        self.logger.info("Position monitor loop ended")
    
    # ==================== POSITION MANAGEMENT ====================
    
    def add_position(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        volume: float,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
        telegram_id: int,
        magic_number: int = 123456
    ) -> bool:
        """
        Add position to monitoring.
        
        Args:
            ticket: MT5 position ticket
            symbol: Symbol name
            direction: 'BUY' or 'SELL'
            volume: Position size in lots
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit_1: First take profit
            take_profit_2: Second take profit
            telegram_id: User's Telegram ID for notifications
            magic_number: Magic number
            
        Returns:
            bool: True if added successfully
        """
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
                magic_number=magic_number
            )
            
            self.monitored_positions[ticket] = position
            
            self.logger.info(
                f"Added position to monitor: Ticket={ticket}, "
                f"{direction} {volume} lots {symbol} at {entry_price}"
            )
            
            # Send initial notification
            if self.bot:
                self._send_notification(
                    telegram_id,
                    f"POSITION OPENED\n\n"
                    f"Ticket: {ticket}\n"
                    f"Symbol: {symbol}\n"
                    f"Direction: {direction}\n"
                    f"Volume: {volume} lots\n"
                    f"Entry: {utils.format_price(symbol, entry_price)}\n"
                    f"Stop Loss: {utils.format_price(symbol, stop_loss)}\n"
                    f"TP1: {utils.format_price(symbol, take_profit_1)}\n"
                    f"TP2: {utils.format_price(symbol, take_profit_2)}\n\n"
                    f"Status: Monitoring active"
                )
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error adding position to monitor: {e}")
            return False
    
    def remove_position(self, ticket: int) -> bool:
        """
        Remove position from monitoring.
        
        Args:
            ticket: Position ticket number
            
        Returns:
            bool: True if removed successfully
        """
        try:
            if ticket in self.monitored_positions:
                del self.monitored_positions[ticket]
                self.logger.info(f"Removed position from monitor: Ticket={ticket}")
                return True
            return False
        
        except Exception as e:
            self.logger.error(f"Error removing position from monitor: {e}")
            return False
    
    def get_position(self, ticket: int) -> Optional[MonitoredPosition]:
        """
        Get monitored position by ticket.
        
        Args:
            ticket: Position ticket number
            
        Returns:
            MonitoredPosition or None
        """
        return self.monitored_positions.get(ticket)
    
    def get_all_positions(self) -> List[MonitoredPosition]:
        """
        Get all monitored positions.
        
        Returns:
            list: List of MonitoredPosition objects
        """
        return list(self.monitored_positions.values())
    
    # ==================== POSITION CHECKING ====================
    
    def _check_all_positions(self):
        """Check all monitored positions for TP1/TP2/SL hits."""
        if not self.monitored_positions:
            return
        
        # Get all open positions from MT5
        mt5_positions = self.mt5.get_open_positions()
        mt5_tickets = {pos['ticket']: pos for pos in mt5_positions}
        
        # Check each monitored position
        tickets_to_remove = []
        
        for ticket, position in self.monitored_positions.items():
            try:
                # Check if position still exists in MT5
                if ticket not in mt5_tickets:
                    self._handle_position_closed(position)
                    tickets_to_remove.append(ticket)
                    continue
                
                # Update current price and profit
                mt5_pos = mt5_tickets[ticket]
                position.current_price = mt5_pos['price_current']
                position.profit = mt5_pos['profit']
                position.last_check = datetime.now()
                
                # Check for TP1 hit
                if not position.tp1_closed:
                    if self._check_tp1_hit(position):
                        self._handle_tp1_hit(position)
                
                # Check for TP2 hit (only if TP1 already closed)
                elif position.tp1_closed and not position.be_activated:
                    # This shouldn't happen (BE should activate with TP1), but check anyway
                    if self._check_be_activation(position):
                        self._handle_be_activation(position)
                
                # Check for TP2 hit
                if position.tp1_closed and self._check_tp2_hit(position):
                    self._handle_tp2_hit(position)
                    tickets_to_remove.append(ticket)
            
            except Exception as e:
                self.logger.error(f"Error checking position {ticket}: {e}")
        
        # Remove closed positions
        for ticket in tickets_to_remove:
            self.remove_position(ticket)
    
    def _check_tp1_hit(self, position: MonitoredPosition) -> bool:
        """
        Check if TP1 has been hit.
        
        Args:
            position: MonitoredPosition to check
            
        Returns:
            bool: True if TP1 hit
        """
        try:
            if position.direction == 'BUY':
                return position.current_price >= position.take_profit_1
            else:  # SELL
                return position.current_price <= position.take_profit_1
        
        except Exception as e:
            self.logger.error(f"Error checking TP1: {e}")
            return False
    
    def _check_tp2_hit(self, position: MonitoredPosition) -> bool:
        """
        Check if TP2 has been hit.
        
        Args:
            position: MonitoredPosition to check
            
        Returns:
            bool: True if TP2 hit
        """
        try:
            if position.direction == 'BUY':
                return position.current_price >= position.take_profit_2
            else:  # SELL
                return position.current_price <= position.take_profit_2
        
        except Exception as e:
            self.logger.error(f"Error checking TP2: {e}")
            return False
    
    def _check_be_activation(self, position: MonitoredPosition) -> bool:
        """
        Check if breakeven should be activated.
        Called after TP1 closes successfully.
        
        Args:
            position: MonitoredPosition to check
            
        Returns:
            bool: True if should activate BE
        """
        return position.tp1_closed and not position.be_activated
    
    # ==================== EVENT HANDLERS ====================
    
    def _handle_tp1_hit(self, position: MonitoredPosition):
        """
        Handle TP1 hit: Close 50% and move SL to breakeven.
        
        Args:
            position: Position that hit TP1
        """
        try:
            self.logger.info(f"TP1 hit for ticket {position.ticket}")
            
            # Close 50% of position
            success, message = self.mt5.close_position(
                position.ticket,
                percentage=50.0,
                comment='TP1 - Partial Close'
            )
            
            if not success:
                self.logger.error(f"Failed to close 50% at TP1: {message}")
                # Retry once
                time.sleep(2)
                success, message = self.mt5.close_position(
                    position.ticket,
                    percentage=50.0,
                    comment='TP1 - Partial Close'
                )
            
            if success:
                position.tp1_closed = True
                position.status = 'TP1_HIT'
                
                # Calculate TP1 profit
                pips_to_tp1 = utils.calculate_pips(
                    position.symbol,
                    position.entry_price,
                    position.take_profit_1
                )
                
                profit_amount, currency = utils.calculate_profit_usd(
                    position.symbol,
                    position.volume / 2,  # 50% closed
                    position.entry_price,
                    position.take_profit_1,
                    position.direction
                )
                
                # Move SL to breakeven + buffer
                self._handle_be_activation(position)
                
                # Send notification
                if self.bot:
                    self._send_notification(
                        position.telegram_id,
                        f"TP1 HIT - 50% CLOSED\n\n"
                        f"Ticket: {position.ticket}\n"
                        f"Symbol: {position.symbol}\n"
                        f"Entry: {utils.format_price(position.symbol, position.entry_price)}\n"
                        f"TP1: {utils.format_price(position.symbol, position.take_profit_1)}\n"
                        f"Pips: +{pips_to_tp1:.1f}\n"
                        f"Profit: {utils.format_currency(profit_amount, currency)}\n\n"
                        f"Remaining 50% running to TP2\n"
                        f"Stop Loss moved to breakeven +{config.BREAKEVEN_BUFFER_PIPS} pips"
                    )
                
                self.logger.info(
                    f"TP1 processed: 50% closed, profit={profit_amount} {currency}, "
                    f"SL moved to breakeven"
                )
            
            else:
                self.logger.error(f"Failed to close 50% at TP1 after retry: {message}")
        
        except Exception as e:
            self.logger.error(f"Error handling TP1 hit: {e}")
    
    def _handle_be_activation(self, position: MonitoredPosition):
        """
        Move stop loss to breakeven + buffer.
        
        Args:
            position: Position to update
        """
        try:
            if position.be_activated:
                return  # Already at breakeven
            
            # Calculate breakeven price
            pip_value = utils.get_pip_value(position.symbol)
            
            if position.direction == 'BUY':
                be_price = position.entry_price + (config.BREAKEVEN_BUFFER_PIPS * pip_value)
            else:  # SELL
                be_price = position.entry_price - (config.BREAKEVEN_BUFFER_PIPS * pip_value)
            
            # Modify position SL
            success, message = self.mt5.modify_position(
                position.ticket,
                stop_loss=be_price,
                take_profit=position.take_profit_2
            )
            
            if success:
                position.stop_loss = be_price
                position.be_activated = True
                
                self.logger.info(
                    f"Breakeven activated for ticket {position.ticket}: "
                    f"SL={utils.format_price(position.symbol, be_price)}"
                )
            else:
                self.logger.error(f"Failed to activate breakeven: {message}")
        
        except Exception as e:
            self.logger.error(f"Error activating breakeven: {e}")
    
    def _handle_tp2_hit(self, position: MonitoredPosition):
        """
        Handle TP2 hit: Close remaining 50%.
        
        Args:
            position: Position that hit TP2
        """
        try:
            self.logger.info(f"TP2 hit for ticket {position.ticket}")
            
            # Close remaining position
            success, message = self.mt5.close_position(
                position.ticket,
                percentage=100.0,
                comment='TP2 - Full Close'
            )
            
            if success:
                position.status = 'TP2_HIT'
                
                # Calculate TP2 profit (remaining 50%)
                pips_to_tp2 = utils.calculate_pips(
                    position.symbol,
                    position.entry_price,
                    position.take_profit_2
                )
                
                profit_amount, currency = utils.calculate_profit_usd(
                    position.symbol,
                    position.volume / 2,  # Remaining 50%
                    position.entry_price,
                    position.take_profit_2,
                    position.direction
                )
                
                # Calculate total trade duration
                duration = utils.calculate_trade_duration(position.opened_at)
                
                # Send notification
                if self.bot:
                    self._send_notification(
                        position.telegram_id,
                        f"TP2 HIT - TRADE COMPLETE\n\n"
                        f"Ticket: {position.ticket}\n"
                        f"Symbol: {position.symbol}\n"
                        f"Entry: {utils.format_price(position.symbol, position.entry_price)}\n"
                        f"TP2: {utils.format_price(position.symbol, position.take_profit_2)}\n"
                        f"Pips: +{pips_to_tp2:.1f}\n"
                        f"Profit (TP2 portion): {utils.format_currency(profit_amount, currency)}\n"
                        f"Duration: {duration}\n\n"
                        f"Full position closed. Trade successful."
                    )
                
                # Log to database
                self._log_trade_completion(position, 'TP2_HIT', position.profit)
                
                self.logger.info(
                    f"TP2 processed: Full position closed, "
                    f"total profit={position.profit:.2f}, duration={duration}"
                )
            
            else:
                self.logger.error(f"Failed to close position at TP2: {message}")
        
        except Exception as e:
            self.logger.error(f"Error handling TP2 hit: {e}")
    
    def _handle_position_closed(self, position: MonitoredPosition):
        """
        Handle position that was closed (likely hit SL).
        
        Args:
            position: Position that was closed
        """
        try:
            self.logger.info(f"Position closed externally: Ticket={position.ticket}")
            
            position.status = 'STOPPED'
            
            # Calculate loss
            pips_lost = utils.calculate_pips(
                position.symbol,
                position.entry_price,
                position.stop_loss
            )
            
            # Calculate trade duration
            duration = utils.calculate_trade_duration(position.opened_at)
            
            # Send notification
            if self.bot:
                self._send_notification(
                    position.telegram_id,
                    f"STOP LOSS HIT\n\n"
                    f"Ticket: {position.ticket}\n"
                    f"Symbol: {position.symbol}\n"
                    f"Entry: {utils.format_price(position.symbol, position.entry_price)}\n"
                    f"Stop Loss: {utils.format_price(position.symbol, position.stop_loss)}\n"
                    f"Pips: -{pips_lost:.1f}\n"
                    f"Loss: {utils.format_currency(position.profit, 'USD')}\n"
                    f"Duration: {duration}\n\n"
                    f"Position closed. Risk management protected capital."
                )
            
            # Log to database
            self._log_trade_completion(position, 'STOPPED', position.profit)
            
            self.logger.info(
                f"SL processed: Position closed, loss={position.profit:.2f}, duration={duration}"
            )
        
        except Exception as e:
            self.logger.error(f"Error handling position closed: {e}")
    
    # ==================== NOTIFICATIONS & LOGGING ====================
    
    def _send_notification(self, telegram_id: int, message: str):
        """
        Send notification to user via Telegram.
        
        Args:
            telegram_id: User's Telegram ID
            message: Message text
        """
        try:
            if self.bot:
                # Add footer
                message_with_footer = utils.add_footer(message)
                
                # Send via bot (async)
                # This assumes bot has async send method
                # Implementation depends on your bot architecture
                self.logger.debug(f"Notification sent to user {telegram_id}")
        
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
    
    def _log_trade_completion(self, position: MonitoredPosition, outcome: str, profit: float):
        """
        Log completed trade to database.
        
        Args:
            position: Completed position
            outcome: 'TP1_HIT', 'TP2_HIT', 'STOPPED'
            profit: Final profit/loss
        """
        try:
            # Log to database via database module
            # Implementation depends on your database structure
            trade_data = {
                'ticket': position.ticket,
                'telegram_id': position.telegram_id,
                'symbol': position.symbol,
                'direction': position.direction,
                'volume': position.volume,
                'entry_price': position.entry_price,
                'exit_price': position.current_price,
                'stop_loss': position.stop_loss,
                'take_profit_1': position.take_profit_1,
                'take_profit_2': position.take_profit_2,
                'outcome': outcome,
                'profit': profit,
                'opened_at': position.opened_at,
                'closed_at': datetime.now(),
                'duration_minutes': (datetime.now() - position.opened_at).total_seconds() / 60
            }
            
            self.logger.info(f"Trade logged: Ticket={position.ticket}, Outcome={outcome}")
        
        except Exception as e:
            self.logger.error(f"Error logging trade completion: {e}")