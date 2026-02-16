"""
NIX TRADES - MT5 Trading Connector
MetaTrader 5 integration with auto-execution, rate limiting, and error recovery
Production-ready, zero errors, zero placeholders
NO EMOJIS - Professional code only
"""

import logging
import time
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from dataclasses import dataclass
import config
import utils

logger = logging.getLogger(__name__)


@dataclass
class MT5RateLimiter:
    """Rate limiter for MT5 API calls to prevent broker throttling."""
    max_calls_per_second: int = 10
    max_calls_per_minute: int = 100
    calls_last_second: List[float] = None
    calls_last_minute: List[float] = None
    
    def __post_init__(self):
        self.calls_last_second = []
        self.calls_last_minute = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        current_time = time.time()
        
        # Clean old calls
        self.calls_last_second = [t for t in self.calls_last_second if current_time - t < 1.0]
        self.calls_last_minute = [t for t in self.calls_last_minute if current_time - t < 60.0]
        
        # Check per-second limit
        if len(self.calls_last_second) >= self.max_calls_per_second:
            sleep_time = 1.0 - (current_time - self.calls_last_second[0])
            if sleep_time > 0:
                logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
                current_time = time.time()
                self.calls_last_second = []
        
        # Check per-minute limit
        if len(self.calls_last_minute) >= self.max_calls_per_minute:
            sleep_time = 60.0 - (current_time - self.calls_last_minute[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit: sleeping {sleep_time:.2f}s (per-minute limit)")
                time.sleep(sleep_time)
                current_time = time.time()
                self.calls_last_minute = []
        
        # Record this call
        self.calls_last_second.append(current_time)
        self.calls_last_minute.append(current_time)


class MT5Connector:
    """
    MetaTrader 5 connector with comprehensive trading functionality.
    Handles connection, data fetching, order execution, and position management.
    """
    
    def __init__(self):
        """Initialize MT5 Connector."""
        self.logger = logging.getLogger(f"{__name__}.MT5Connector")
        self.connected = False
        self.account_info = None
        self.rate_limiter = MT5RateLimiter()
        self.logger.info("MT5 Connector initialized")
    
    # ==================== CONNECTION MANAGEMENT ====================
    
    def connect(
        self,
        login: int,
        password: str,
        server: str,
        timeout: int = config.MT5_TIMEOUT
    ) -> Tuple[bool, str]:
        """
        Connect to MT5 terminal with triple-retry logic.
        
        Args:
            login: MT5 account login
            password: MT5 account password
            server: Broker server name
            timeout: Connection timeout in milliseconds
            
        Returns:
            tuple: (success, message)
        """
        for attempt in range(config.MT5_RETRY_ATTEMPTS):
            try:
                # Initialize MT5
                if not mt5.initialize():
                    error = mt5.last_error()
                    self.logger.error(f"MT5 initialization failed: {error}")
                    
                    if attempt < config.MT5_RETRY_ATTEMPTS - 1:
                        time.sleep(config.MT5_RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        return False, f"MT5 initialization failed: {error}"
                
                # Login to account
                if not mt5.login(login, password=password, server=server, timeout=timeout):
                    error = mt5.last_error()
                    self.logger.error(f"MT5 login failed: {error}")
                    
                    if attempt < config.MT5_RETRY_ATTEMPTS - 1:
                        time.sleep(config.MT5_RETRY_DELAY * (attempt + 1))
                        mt5.shutdown()
                        continue
                    else:
                        mt5.shutdown()
                        return False, f"Login failed: {error}"
                
                # Get account info
                account_info = mt5.account_info()
                if account_info is None:
                    error = mt5.last_error()
                    self.logger.error(f"Failed to get account info: {error}")
                    
                    if attempt < config.MT5_RETRY_ATTEMPTS - 1:
                        time.sleep(config.MT5_RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        return False, f"Failed to get account info: {error}"
                
                # Success
                self.connected = True
                self.account_info = account_info._asdict()
                
                self.logger.info(
                    f"MT5 connected: Account {login}, "
                    f"Balance: {self.account_info['balance']} {self.account_info['currency']}"
                )
                
                # Add monitored symbols to Market Watch
                self._add_symbols_to_market_watch(config.MONITORED_SYMBOLS)
                
                return True, "Connected successfully"
            
            except Exception as e:
                self.logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                
                if attempt < config.MT5_RETRY_ATTEMPTS - 1:
                    time.sleep(config.MT5_RETRY_DELAY * (attempt + 1))
                else:
                    return False, f"Connection error: {str(e)}"
        
        return False, "Connection failed after all retries"
    
    def disconnect(self) -> bool:
        """
        Disconnect from MT5 terminal.
        
        Returns:
            bool: True if disconnected successfully
        """
        try:
            if self.connected:
                mt5.shutdown()
                self.connected = False
                self.account_info = None
                self.logger.info("MT5 disconnected")
                return True
            return True
        
        except Exception as e:
            self.logger.error(f"Error disconnecting: {e}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if MT5 is connected.
        
        Returns:
            bool: True if connected
        """
        try:
            if not self.connected:
                return False
            
            # Verify connection is still alive
            account_info = mt5.account_info()
            if account_info is None:
                self.connected = False
                return False
            
            return True
        
        except:
            self.connected = False
            return False
    
    def _add_symbols_to_market_watch(self, symbols: List[str]) -> None:
        """
        Add symbols to Market Watch for real-time quotes.
        
        Args:
            symbols: List of symbol names
        """
        try:
            for symbol in symbols:
                self.rate_limiter.wait_if_needed()
                
                # Try standard symbol first
                if not mt5.symbol_select(symbol, True):
                    # Try with common suffixes
                    found = False
                    for suffix in config.SYMBOL_SUFFIXES:
                        broker_symbol = f"{symbol}{suffix}"
                        if mt5.symbol_select(broker_symbol, True):
                            self.logger.info(f"Added {broker_symbol} to Market Watch")
                            found = True
                            break
                    
                    if not found:
                        self.logger.warning(f"Could not add {symbol} to Market Watch")
                else:
                    self.logger.info(f"Added {symbol} to Market Watch")
        
        except Exception as e:
            self.logger.error(f"Error adding symbols to Market Watch: {e}")
    
    # ==================== ACCOUNT OPERATIONS ====================
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current account information with rate limiting.
        
        Returns:
            dict: Account info or None if failed
        """
        try:
            if not self.is_connected():
                return None
            
            self.rate_limiter.wait_if_needed()
            
            account_info = mt5.account_info()
            if account_info is None:
                return None
            
            info_dict = account_info._asdict()
            self.account_info = info_dict
            
            return info_dict
        
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None
    
    def get_account_balance(self) -> Tuple[Optional[float], Optional[str]]:
        """
        Get account balance and currency.
        
        Returns:
            tuple: (balance, currency) or (None, None)
        """
        info = self.get_account_info()
        if info:
            return info['balance'], info['currency']
        return None, None
    
    def get_account_equity(self) -> Optional[float]:
        """
        Get account equity (balance + floating profit/loss).
        
        Returns:
            float: Equity or None
        """
        info = self.get_account_info()
        return info['equity'] if info else None
    
    def get_exchange_rates(self) -> Dict[str, float]:
        """
        Get exchange rates for all major currency pairs for lot size calculations.
        
        Returns:
            dict: Exchange rates (e.g., {'EURUSD': 1.0850, 'GBPUSD': 1.2650})
        """
        rates = {}
        
        try:
            # Major pairs
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']
            
            for pair in major_pairs:
                self.rate_limiter.wait_if_needed()
                
                symbol_info = self.get_symbol_info(pair)
                if symbol_info and symbol_info['bid'] > 0:
                    rates[pair] = symbol_info['bid']
        
        except Exception as e:
            self.logger.error(f"Error getting exchange rates: {e}")
        
        return rates
    
    # ==================== SYMBOL OPERATIONS ====================
    
    def normalize_symbol(self, standard_symbol: str) -> Optional[str]:
        """
        Find broker's actual symbol name from standard symbol.
        
        Args:
            standard_symbol: Standard symbol (e.g., 'EURUSD')
            
        Returns:
            str: Broker's symbol or None if not found
        """
        try:
            self.rate_limiter.wait_if_needed()
            
            # Try standard symbol first
            symbol_info = mt5.symbol_info(standard_symbol)
            if symbol_info is not None:
                return standard_symbol
            
            # Try with suffixes
            for suffix in config.SYMBOL_SUFFIXES:
                broker_symbol = f"{standard_symbol}{suffix}"
                symbol_info = mt5.symbol_info(broker_symbol)
                if symbol_info is not None:
                    self.logger.info(f"Normalized {standard_symbol} to {broker_symbol}")
                    return broker_symbol
            
            self.logger.warning(f"Symbol {standard_symbol} not found on broker")
            return None
        
        except Exception as e:
            self.logger.error(f"Error normalizing symbol: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol information with rate limiting.
        
        Args:
            symbol: Symbol name
            
        Returns:
            dict: Symbol info or None
        """
        try:
            self.rate_limiter.wait_if_needed()
            
            broker_symbol = self.normalize_symbol(symbol)
            if not broker_symbol:
                return None
            
            symbol_info = mt5.symbol_info(broker_symbol)
            if symbol_info is None:
                return None
            
            # Get current tick for prices
            tick = mt5.symbol_info_tick(broker_symbol)
            
            info_dict = symbol_info._asdict()
            
            if tick:
                info_dict['bid'] = tick.bid
                info_dict['ask'] = tick.ask
                info_dict['last'] = tick.last
                info_dict['time'] = tick.time
            
            return info_dict
        
        except Exception as e:
            self.logger.error(f"Error getting symbol info: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Get current bid and ask prices with rate limiting.
        
        Args:
            symbol: Symbol name
            
        Returns:
            tuple: (bid, ask) or (None, None)
        """
        try:
            self.rate_limiter.wait_if_needed()
            
            broker_symbol = self.normalize_symbol(symbol)
            if not broker_symbol:
                return None, None
            
            tick = mt5.symbol_info_tick(broker_symbol)
            if tick is None:
                return None, None
            
            return tick.bid, tick.ask
        
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None, None
    
    # ==================== HISTORICAL DATA ====================
    
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        bars: int = 1000
    ) -> Optional[List[Dict]]:
        """
        Get historical OHLCV data with rate limiting.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
            bars: Number of bars to fetch
            
        Returns:
            list: List of OHLCV dicts or None
        """
        try:
            self.rate_limiter.wait_if_needed()
            
            broker_symbol = self.normalize_symbol(symbol)
            if not broker_symbol:
                return None
            
            # Map timeframe string to MT5 constant
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1,
                'W1': mt5.TIMEFRAME_W1,
                'MN1': mt5.TIMEFRAME_MN1,
            }
            
            mt5_timeframe = timeframe_map.get(timeframe)
            if not mt5_timeframe:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            # Fetch data
            rates = mt5.copy_rates_from_pos(broker_symbol, mt5_timeframe, 0, bars)
            
            if rates is None or len(rates) == 0:
                self.logger.error(f"No data for {broker_symbol} {timeframe}")
                return None
            
            # Convert to list of dicts
            data = []
            for rate in rates:
                data.append({
                    'time': datetime.fromtimestamp(rate['time']),
                    'open': rate['open'],
                    'high': rate['high'],
                    'low': rate['low'],
                    'close': rate['close'],
                    'volume': rate['tick_volume']
                })
            
            self.logger.info(f"Fetched {len(data)} bars for {broker_symbol} {timeframe}")
            return data
        
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return None
    
    # ==================== ORDER EXECUTION ====================
    
    def place_order(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        order_type: str = 'MARKET',
        comment: str = 'Nix Trades',
        magic_number: int = 123456,
        expiration: Optional[datetime] = None
    ) -> Tuple[bool, Optional[int], str]:
        """
        Place trading order with triple-retry logic and rate limiting.
        
        Args:
            symbol: Symbol name
            direction: 'BUY' or 'SELL'
            lot_size: Position size in lots
            entry_price: Entry price (for limit/stop orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            order_type: 'MARKET', 'LIMIT', or 'STOP'
            comment: Order comment
            magic_number: Magic number for identification
            expiration: Order expiration time (for pending orders)
            
        Returns:
            tuple: (success, ticket_number, message)
        """
        for attempt in range(config.MT5_RETRY_ATTEMPTS):
            try:
                if not self.is_connected():
                    return False, None, "Not connected to MT5"
                
                self.rate_limiter.wait_if_needed()
                
                broker_symbol = self.normalize_symbol(symbol)
                if not broker_symbol:
                    return False, None, f"Symbol {symbol} not found"
                
                # Get current prices
                bid, ask = self.get_current_price(broker_symbol)
                if not bid or not ask:
                    return False, None, "Failed to get current prices"
                
                # Determine order type
                if direction == 'BUY':
                    trade_type = mt5.ORDER_TYPE_BUY if order_type == 'MARKET' else (
                        mt5.ORDER_TYPE_BUY_LIMIT if order_type == 'LIMIT' else mt5.ORDER_TYPE_BUY_STOP
                    )
                    price = ask if order_type == 'MARKET' else entry_price
                else:  # SELL
                    trade_type = mt5.ORDER_TYPE_SELL if order_type == 'MARKET' else (
                        mt5.ORDER_TYPE_SELL_LIMIT if order_type == 'LIMIT' else mt5.ORDER_TYPE_SELL_STOP
                    )
                    price = bid if order_type == 'MARKET' else entry_price
                
                # Get symbol info for lot size validation
                symbol_info = mt5.symbol_info(broker_symbol)
                if symbol_info is None:
                    return False, None, "Failed to get symbol info"
                
                # Validate lot size
                if lot_size < symbol_info.volume_min:
                    lot_size = symbol_info.volume_min
                if lot_size > symbol_info.volume_max:
                    lot_size = symbol_info.volume_max
                
                # Round lot size to step
                lot_step = symbol_info.volume_step
                lot_size = round(lot_size / lot_step) * lot_step
                
                # Build request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL if order_type == 'MARKET' else mt5.TRADE_ACTION_PENDING,
                    "symbol": broker_symbol,
                    "volume": lot_size,
                    "type": trade_type,
                    "price": price,
                    "deviation": 20,
                    "magic": magic_number,
                    "comment": comment,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                # Add SL/TP if provided
                if stop_loss:
                    request['sl'] = stop_loss
                if take_profit:
                    request['tp'] = take_profit
                
                # Add expiration for pending orders
                if order_type != 'MARKET' and expiration:
                    request['type_time'] = mt5.ORDER_TIME_SPECIFIED
                    request['expiration'] = int(expiration.timestamp())
                
                # Send order
                result = mt5.order_send(request)
                
                if result is None:
                    error = mt5.last_error()
                    self.logger.error(f"Order send failed (attempt {attempt + 1}): {error}")
                    
                    if attempt < config.MT5_RETRY_ATTEMPTS - 1:
                        time.sleep(config.MT5_RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        return False, None, f"Order failed: {error}"
                
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    self.logger.error(
                        f"Order rejected (attempt {attempt + 1}): "
                        f"retcode={result.retcode}, comment={result.comment}"
                    )
                    
                    if attempt < config.MT5_RETRY_ATTEMPTS - 1:
                        time.sleep(config.MT5_RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        return False, None, f"Order rejected: {result.comment}"
                
                # Success
                ticket = result.order if order_type != 'MARKET' else result.deal
                
                self.logger.info(
                    f"Order placed: {direction} {lot_size} lots {broker_symbol} "
                    f"at {price}, ticket={ticket}"
                )
                
                return True, ticket, "Order placed successfully"
            
            except Exception as e:
                self.logger.error(f"Order placement error (attempt {attempt + 1}): {e}")
                
                if attempt < config.MT5_RETRY_ATTEMPTS - 1:
                    time.sleep(config.MT5_RETRY_DELAY * (attempt + 1))
                else:
                    return False, None, f"Order error: {str(e)}"
        
        return False, None, "Order failed after all retries"
    
    # ==================== POSITION MANAGEMENT ====================
    
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open positions with rate limiting.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            list: List of position dicts
        """
        try:
            if not self.is_connected():
                return []
            
            self.rate_limiter.wait_if_needed()
            
            if symbol:
                broker_symbol = self.normalize_symbol(symbol)
                if not broker_symbol:
                    return []
                positions = mt5.positions_get(symbol=broker_symbol)
            else:
                positions = mt5.positions_get()
            
            if positions is None or len(positions) == 0:
                return []
            
            position_list = []
            for pos in positions:
                position_list.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'commission': pos.commission,
                    'magic': pos.magic,
                    'comment': pos.comment,
                    'time': datetime.fromtimestamp(pos.time)
                })
            
            return position_list
        
        except Exception as e:
            self.logger.error(f"Error getting open positions: {e}")
            return []
    
    def modify_position(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Modify position SL/TP with triple-retry logic and rate limiting.
        
        Args:
            ticket: Position ticket number
            stop_loss: New stop loss price
            take_profit: New take profit price
            
        Returns:
            tuple: (success, message)
        """
        for attempt in range(config.MT5_RETRY_ATTEMPTS):
            try:
                if not self.is_connected():
                    return False, "Not connected to MT5"
                
                self.rate_limiter.wait_if_needed()
                
                # Get position
                position = mt5.positions_get(ticket=ticket)
                if not position or len(position) == 0:
                    return False, f"Position {ticket} not found"
                
                position = position[0]
                
                # Build request
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": position.symbol,
                    "position": ticket,
                    "sl": stop_loss if stop_loss else position.sl,
                    "tp": take_profit if take_profit else position.tp,
                }
                
                # Send modification
                result = mt5.order_send(request)
                
                if result is None:
                    error = mt5.last_error()
                    self.logger.error(f"Position modify failed (attempt {attempt + 1}): {error}")
                    
                    if attempt < config.MT5_RETRY_ATTEMPTS - 1:
                        time.sleep(config.MT5_RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        return False, f"Modify failed: {error}"
                
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    self.logger.error(
                        f"Position modify rejected (attempt {attempt + 1}): "
                        f"retcode={result.retcode}, comment={result.comment}"
                    )
                    
                    if attempt < config.MT5_RETRY_ATTEMPTS - 1:
                        time.sleep(config.MT5_RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        return False, f"Modify rejected: {result.comment}"
                
                # Success
                self.logger.info(f"Position {ticket} modified: SL={stop_loss}, TP={take_profit}")
                return True, "Position modified successfully"
            
            except Exception as e:
                self.logger.error(f"Position modify error (attempt {attempt + 1}): {e}")
                
                if attempt < config.MT5_RETRY_ATTEMPTS - 1:
                    time.sleep(config.MT5_RETRY_DELAY * (attempt + 1))
                else:
                    return False, f"Modify error: {str(e)}"
        
        return False, "Modify failed after all retries"
    
    def close_position(
        self,
        ticket: int,
        percentage: float = 100.0,
        comment: str = 'Nix Trades Close'
    ) -> Tuple[bool, str]:
        """
        Close position (full or partial) with triple-retry logic and rate limiting.
        
        Args:
            ticket: Position ticket number
            percentage: Percentage to close (50.0 for partial, 100.0 for full)
            comment: Close comment
            
        Returns:
            tuple: (success, message)
        """
        for attempt in range(config.MT5_RETRY_ATTEMPTS):
            try:
                if not self.is_connected():
                    return False, "Not connected to MT5"
                
                self.rate_limiter.wait_if_needed()
                
                # Get position
                position = mt5.positions_get(ticket=ticket)
                if not position or len(position) == 0:
                    return False, f"Position {ticket} not found"
                
                position = position[0]
                
                # Calculate volume to close
                volume_to_close = position.volume * (percentage / 100.0)
                
                # Get symbol info for rounding
                symbol_info = mt5.symbol_info(position.symbol)
                if symbol_info:
                    volume_to_close = round(volume_to_close / symbol_info.volume_step) * symbol_info.volume_step
                
                # Determine close type (opposite of open)
                close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                
                # Get current price
                tick = mt5.symbol_info_tick(position.symbol)
                if not tick:
                    return False, "Failed to get current price"
                
                close_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
                
                # Build request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "volume": volume_to_close,
                    "type": close_type,
                    "position": ticket,
                    "price": close_price,
                    "deviation": 20,
                    "magic": position.magic,
                    "comment": comment,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                # Send close order
                result = mt5.order_send(request)
                
                if result is None:
                    error = mt5.last_error()
                    self.logger.error(f"Position close failed (attempt {attempt + 1}): {error}")
                    
                    if attempt < config.MT5_RETRY_ATTEMPTS - 1:
                        time.sleep(config.MT5_RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        return False, f"Close failed: {error}"
                
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    self.logger.error(
                        f"Position close rejected (attempt {attempt + 1}): "
                        f"retcode={result.retcode}, comment={result.comment}"
                    )
                    
                    if attempt < config.MT5_RETRY_ATTEMPTS - 1:
                        time.sleep(config.MT5_RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        return False, f"Close rejected: {result.comment}"
                
                # Success
                close_type_str = "PARTIAL" if percentage < 100 else "FULL"
                self.logger.info(
                    f"Position {ticket} closed ({close_type_str}): "
                    f"{volume_to_close} lots at {close_price}"
                )
                
                return True, f"Position closed successfully ({percentage}%)"
            
            except Exception as e:
                self.logger.error(f"Position close error (attempt {attempt + 1}): {e}")
                
                if attempt < config.MT5_RETRY_ATTEMPTS - 1:
                    time.sleep(config.MT5_RETRY_DELAY * (attempt + 1))
                else:
                    return False, f"Close error: {str(e)}"
        
        return False, "Close failed after all retries"
    
    def cancel_pending_order(self, ticket: int) -> Tuple[bool, str]:
        """
        Cancel pending order with rate limiting.
        
        Args:
            ticket: Order ticket number
            
        Returns:
            tuple: (success, message)
        """
        try:
            if not self.is_connected():
                return False, "Not connected to MT5"
            
            self.rate_limiter.wait_if_needed()
            
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket,
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                return False, f"Cancel failed: {error}"
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return False, f"Cancel rejected: {result.comment}"
            
            self.logger.info(f"Pending order {ticket} cancelled")
            return True, "Order cancelled successfully"
        
        except Exception as e:
            self.logger.error(f"Order cancel error: {e}")
            return False, f"Cancel error: {str(e)}"