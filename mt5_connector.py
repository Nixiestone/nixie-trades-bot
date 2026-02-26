"""
NIX TRADES - MT5 Connector
Role: Lead Architect + Quant Software Engineer + Security Engineer

HTTP client that communicates with the MT5 Worker Service running on Windows.
This module runs on the Linux bot server. It does NOT import MetaTrader5 directly.

Changes in this version:
  - place_order now accepts sl_pips and risk_percent; lot_size=0.0 tells
    the worker to calculate lot size server-side using the correct formula
  - All HTTP calls use exponential backoff retry (4 attempts: 0s, 1s, 3s, 9s)
  - Credential scrubbing: passwords are never logged
  - get_current_price returns (bid, ask) tuple, called synchronously
    so the async scheduler can use run_in_executor

NO EMOJIS - Enterprise code only
"""

import logging
import time
from typing import Optional, Dict, Tuple, Any

import requests

import config
import database as db

logger = logging.getLogger(__name__)

# Retry configuration
_RETRY_DELAYS = (0, 1, 3, 9)  # seconds before each attempt (4 total)


class MT5Connector:
    """
    Client-side connector that translates bot requests into HTTP calls
    to the MT5 Worker Service running on a Windows VPS.

    Each method retrieves the user's encrypted MT5 credentials from the
    database, injects them into the request, and returns the result.
    This design allows unlimited simultaneous users because the worker
    holds the per-process MT5 lock, not this module.
    """

    def __init__(self):
        self.logger   = logging.getLogger(f"{__name__}.MT5Connector")
        self.base_url = config.MT5_WORKER_URL.rstrip('/')
        self.api_key  = config.MT5_WORKER_API_KEY
        self.timeout  = config.MT5_TIMEOUT
        self.logger.info("MT5 Connector initialised. Worker URL: %s", self.base_url)

    # ==================== INTERNAL HELPERS ====================

    def _headers(self) -> Dict[str, str]:
        return {
            'Content-Type': 'application/json',
            'X-API-Key':    self.api_key,
        }

    def _post(self, endpoint: str, payload: dict) -> Tuple[bool, Any]:
        """
        POST to the MT5 worker with exponential backoff retry.

        Attempts: 4
        Delays (before each attempt): 0s, 1s, 3s, 9s

        Passwords in the payload are never written to logs.

        Returns:
            (True, response_dict) on success
            (False, error_string) on permanent failure
        """
        url = f"{self.base_url}{endpoint}"

        # Build a safe payload for logging (password redacted)
        safe_payload = {
            k: ('***REDACTED***' if 'password' in k.lower() else v)
            for k, v in payload.items()
        }

        for attempt, delay in enumerate(_RETRY_DELAYS):
            if delay > 0:
                time.sleep(delay)

            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=(5, 30),   # 5s connect timeout, 30s read timeout
                )

                if response.status_code == 401:
                    return False, "Execution service authentication failed. Check MT5_WORKER_API_KEY."

                if response.status_code == 413:
                    return False, "Request rejected: payload too large."

                if response.status_code == 503:
                    self.logger.warning(
                        "Worker busy (attempt %d/4) endpoint=%s",
                        attempt + 1, endpoint
                    )
                    continue

                data = response.json()

                if data.get('success'):
                    return True, data

                error_msg = data.get('error', 'Unknown error from execution service.')
                self.logger.error("Worker error on %s: %s", endpoint, error_msg)
                return False, error_msg

            except requests.exceptions.ConnectionError:
                self.logger.warning(
                    "Worker unreachable at %s (attempt %d/4)", url, attempt + 1
                )

            except requests.exceptions.Timeout:
                self.logger.warning(
                    "Worker timeout on %s (attempt %d/4)", endpoint, attempt + 1
                )

            except Exception as e:
                self.logger.error("Unexpected error calling %s: %s", endpoint, e)
                return False, str(e)

        self.logger.error(
            "Worker permanently unreachable after 4 attempts: %s", endpoint
        )
        return False, "The execution service is temporarily unavailable. Please try again in a few minutes."

    def _get(self, endpoint: str, params: dict = None) -> Tuple[bool, Any]:
        """
        GET from the MT5 worker with exponential backoff retry.
        """
        url = f"{self.base_url}{endpoint}"

        for attempt, delay in enumerate(_RETRY_DELAYS):
            if delay > 0:
                time.sleep(delay)

            try:
                response = requests.get(
                    url,
                    headers=self._headers(),
                    params=params or {},
                    timeout=self.timeout,
                )

                if response.status_code == 401:
                    return False, "Execution service authentication failed."

                if response.status_code == 503:
                    self.logger.warning(
                        "Worker busy on GET %s (attempt %d/4)", endpoint, attempt + 1
                    )
                    continue

                data = response.json()
                if data.get('success'):
                    return True, data
                return False, data.get('error', 'Unknown error.')

            except requests.exceptions.ConnectionError:
                self.logger.warning(
                    "Worker unreachable (GET) at %s (attempt %d/4)", endpoint, attempt + 1
                )

            except requests.exceptions.Timeout:
                self.logger.warning(
                    "Worker GET timeout on %s (attempt %d/4)", endpoint, attempt + 1
                )

            except Exception as e:
                self.logger.error("Unexpected GET error on %s: %s", endpoint, e)
                return False, str(e)

        return False, "The execution service is temporarily unavailable."

    def _get_credentials(self, telegram_id: int) -> Optional[dict]:
        """Retrieve decrypted MT5 credentials for a user from the database."""
        creds = db.get_mt5_credentials(telegram_id)
        if not creds:
            self.logger.warning("No MT5 credentials found for user %d", telegram_id)
        return creds

    # ==================== CONNECTIVITY ====================

    def is_worker_reachable(self) -> bool:
        """Check if the MT5 Worker Service is online."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self._headers(),
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def is_connected(self) -> bool:
        """Alias for is_worker_reachable (called by scheduler)."""
        return self.is_worker_reachable()

    # ==================== CONNECTION VERIFICATION ====================

    def verify_credentials(
        self,
        telegram_id: int,
        login: int,
        password: str,
        server: str,
    ) -> Tuple[bool, Any]:
        """
        Verify MT5 credentials and return account information.
        Called during /connect_mt5 to validate before saving.

        Passwords are never logged.

        Returns:
            (True, account_info_dict) on success
            (False, error_string) on failure
        """
        payload = {
            'login':    login,
            'password': password,
            'server':   server,
        }
        success, result = self._post('/connect', payload)
        return success, result

    # ==================== MARKET DATA ====================

    def get_current_price(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Return current (bid, ask) for a symbol.
        Returns (None, None) if the worker is unreachable or symbol not found.
        This method is synchronous so it can be used via run_in_executor.
        """
        success, result = self._get('/tick', params={'symbol': symbol})
        if success and isinstance(result, dict):
            return result.get('bid'), result.get('ask')
        return None, None

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        bars: int = 200,
    ) -> Optional[list]:
        """
        Fetch historical OHLCV candle data from the worker.

        Returns list of dicts with keys: time, open, high, low, close, volume
        Returns None on failure.
        """
        payload = {
            'symbol':    symbol,
            'timeframe': timeframe,
            'bars':      bars,
        }
        success, result = self._post('/candles', payload)
        if success and isinstance(result, dict):
            return result.get('candles')
        self.logger.warning(
            "Failed to get historical data for %s %s: %s", symbol, timeframe, result
        )
        return None

    def get_exchange_rates(self) -> dict:
        """
        Return currency exchange rates from the worker.
        Falls back to empty dict which causes lot size to default
        to minimum (safe fallback).
        """
        success, result = self._get('/exchange_rates')
        if success and isinstance(result, dict):
            return result.get('rates', {})
        return {}

    # ==================== ORDER EXECUTION ====================

    def place_order(
        self,
        telegram_id: int,
        symbol: str,
        direction: str,
        lot_size: float,
        stop_loss: float,
        take_profit: float,
        order_type: str = 'MARKET',
        entry_price: Optional[float] = None,
        comment: str = 'Nix Trades',
        expiry_minutes: int = 120,
        sl_pips: float = 0.0,
        risk_percent: float = 1.0,
    ) -> Tuple[bool, Optional[int], str]:
        """
        Place a trade order on the user's MT5 account.

        When lot_size is 0.0, the worker calculates it server-side using:
            lot_size = risk_amount / (price_distance / tick_size * tick_value)

        Args:
            telegram_id:    Telegram user ID (used to look up credentials)
            symbol:         Trading symbol (e.g. 'EURUSD')
            direction:      'BUY' or 'SELL'
            lot_size:       0.0 = let worker calculate; >0 = use exact value
            stop_loss:      Stop loss price
            take_profit:    Take profit price (TP2 for the initial order)
            order_type:     'MARKET', 'LIMIT', or 'STOP'
            entry_price:    Entry price for pending orders
            comment:        Order comment (max 31 chars for MT5)
            expiry_minutes: Minutes until pending order expires
            sl_pips:        Stop-loss distance in pips (used for lot calculation)
            risk_percent:   Risk per trade as a percentage

        Returns:
            Tuple[bool, Optional[int], str]: (success, ticket_number, message)
        """
        creds = self._get_credentials(telegram_id)
        if not creds:
            return False, None, "MT5 credentials not found. Please use /connect_mt5 first."

        expiry_hours = max(1, expiry_minutes // 60)

        payload = {
            'login':        creds['login'],
            'password':     creds['password'],   # scrubbed in worker logs
            'server':       creds['server'],
            'symbol':       symbol,
            'direction':    direction,
            'entry':        entry_price if entry_price is not None else 0.0,
            'sl':           stop_loss,
            'tp1':          take_profit,
            'tp2':          take_profit,          # bot sets tp2 same as tp1; position monitor updates
            'sl_pips':      sl_pips,
            'risk_percent': risk_percent,
            'lot_size':     lot_size,
            'order_type':   order_type,
            'comment':      comment[:31],
            'expiry_hours': expiry_hours,
        }

        success, result = self._post('/execute', payload)

        if success and isinstance(result, dict):
            ticket = result.get('order') or result.get('deal')
            return True, ticket, "Order placed successfully."

        error_msg = result if isinstance(result, str) else "Order placement failed."
        return False, None, error_msg

    # ==================== POSITION MANAGEMENT ====================

    def get_positions(self, telegram_id: int) -> Tuple[bool, list]:
        """
        Return all open positions for a user's account.

        Returns:
            (True, positions_list) or (False, [])
        """
        creds = self._get_credentials(telegram_id)
        if not creds:
            return False, []

        payload = {
            'login':    creds['login'],
            'password': creds['password'],
            'server':   creds['server'],
        }
        success, result = self._post('/positions', payload)
        if success and isinstance(result, dict):
            return True, result.get('positions', [])
        return False, []

    def close_partial_position(
        self,
        telegram_id: int,
        ticket: int,
        close_pct: float,
    ) -> Tuple[bool, str]:
        """
        Close a percentage of an open position.

        Args:
            telegram_id: Telegram user ID
            ticket:      MT5 position ticket number
            close_pct:   Fraction to close (0.01 to 1.0)

        Returns:
            (True, message) or (False, error_message)
        """
        creds = self._get_credentials(telegram_id)
        if not creds:
            return False, "MT5 credentials not found."

        payload = {
            'login':     creds['login'],
            'password':  creds['password'],
            'server':    creds['server'],
            'ticket':    ticket,
            'close_pct': close_pct,
        }
        success, result = self._post('/close_partial', payload)
        if success:
            return True, f"Closed {close_pct * 100:.0f}% of position {ticket}."
        return False, result if isinstance(result, str) else "Partial close failed."

    def modify_stop_loss(
        self,
        telegram_id: int,
        ticket: int,
        new_sl: float,
    ) -> Tuple[bool, str]:
        """
        Move the stop-loss of an open position.

        Args:
            telegram_id: Telegram user ID
            ticket:      MT5 position ticket number
            new_sl:      New stop-loss price

        Returns:
            (True, message) or (False, error_message)
        """
        creds = self._get_credentials(telegram_id)
        if not creds:
            return False, "MT5 credentials not found."

        payload = {
            'login':    creds['login'],
            'password': creds['password'],
            'server':   creds['server'],
            'ticket':   ticket,
            'new_sl':   new_sl,
        }
        success, result = self._post('/modify_sl', payload)
        if success:
            return True, f"Stop loss moved to {new_sl:.5f} on position {ticket}."
        return False, result if isinstance(result, str) else "Stop loss modification failed."

    def get_account_info(self, telegram_id: int) -> Tuple[bool, dict]:
        """
        Return full account information for a user.

        Returns:
            (True, account_dict) or (False, {})
        """
        creds = self._get_credentials(telegram_id)
        if not creds:
            return False, {}

        payload = {
            'login':    creds['login'],
            'password': creds['password'],
            'server':   creds['server'],
        }
        success, result = self._post('/account', payload)
        if success and isinstance(result, dict):
            return True, result
        return False, {}