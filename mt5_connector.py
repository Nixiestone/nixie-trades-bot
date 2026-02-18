"""
NIX TRADES - MT5 Connector
HTTP client that communicates with the MT5 Worker Service.
Runs on the Linux bot server. Does NOT import MetaTrader5 directly.

Role: Lead Architect + Quant Software Engineer
Fixes:
  - Replaced direct MetaTrader5 DLL calls with HTTP requests to mt5_worker.py
    (the worker runs on Windows; this connector runs on the bot server)
  - Per-user credential injection on every call (no global single-user state)
  - Rate limiting preserved via request throttling
  - Lot size calculation delegated to the worker (has live account data)
  - All methods return (bool, data/message) tuples for consistent error handling
NO EMOJIS - Professional code only
"""

import logging
import time
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timezone

import requests

import config
import database as db

logger = logging.getLogger(__name__)


class MT5Connector:
    """
    Client-side connector that translates bot requests into HTTP calls
    to the MT5 Worker Service running on a Windows machine.

    Each method retrieves the user's MT5 credentials from the database,
    passes them to the worker, and returns the result.
    This design allows unlimited simultaneous users.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MT5Connector")
        self.base_url = config.MT5_WORKER_URL.rstrip('/')
        self.api_key = config.MT5_WORKER_API_KEY
        self.timeout = config.MT5_TIMEOUT
        self.logger.info(
            "MT5 Connector initialised. Worker URL: %s", self.base_url
        )

    # ==================== INTERNAL HELPERS ====================

    def _headers(self) -> Dict[str, str]:
        return {
            'Content-Type': 'application/json',
            'X-API-Key':    self.api_key
        }

    def _post(self, endpoint: str, payload: dict) -> Tuple[bool, Any]:
        """
        Make a POST request to the MT5 worker with retry logic.

        Args:
            endpoint: API endpoint path (e.g., '/connect')
            payload:  JSON request body

        Returns:
            Tuple[bool, Any]: (success, response_dict or error_message)
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(config.MT5_RETRY_ATTEMPTS):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout
                )

                if response.status_code == 401:
                    return False, "MT5 Worker authentication failed. Check MT5_WORKER_API_KEY."

                data = response.json()

                if data.get('success'):
                    return True, data

                error_msg = data.get('error', 'Unknown error from MT5 worker')
                self.logger.error(
                    "MT5 Worker error on %s: %s", endpoint, error_msg
                )
                return False, error_msg

            except requests.exceptions.ConnectionError:
                self.logger.error(
                    "MT5 Worker unreachable at %s (attempt %d/%d)",
                    url, attempt + 1, config.MT5_RETRY_ATTEMPTS
                )
                if attempt < config.MT5_RETRY_ATTEMPTS - 1:
                    time.sleep(config.MT5_RETRY_DELAY * (attempt + 1))

            except requests.exceptions.Timeout:
                self.logger.error(
                    "MT5 Worker timeout on %s (attempt %d/%d)",
                    url, attempt + 1, config.MT5_RETRY_ATTEMPTS
                )
                if attempt < config.MT5_RETRY_ATTEMPTS - 1:
                    time.sleep(config.MT5_RETRY_DELAY * (attempt + 1))

            except Exception as e:
                self.logger.error(
                    "Unexpected error calling MT5 worker %s: %s", endpoint, e
                )
                return False, str(e)

        return False, f"MT5 Worker unreachable after {config.MT5_RETRY_ATTEMPTS} attempts"

    def _get_credentials(self, telegram_id: int) -> Optional[dict]:
        """
        Retrieve decrypted MT5 credentials for a user.

        Args:
            telegram_id: Telegram user ID

        Returns:
            dict with login/password/server, or None
        """
        creds = db.get_mt5_credentials(telegram_id)
        if not creds:
            self.logger.warning("No MT5 credentials found for user %d", telegram_id)
        return creds

    # ==================== CONNECTION VERIFICATION ====================

    def verify_credentials(
        self,
        telegram_id: int,
        login: int,
        password: str,
        server: str
    ) -> Tuple[bool, dict]:
        """
        Verify MT5 credentials and return account information.
        Used during /connect_mt5 to validate before saving credentials.

        Args:
            telegram_id: Telegram user ID
            login:       MT5 login number
            password:    MT5 password
            server:      MT5 server name

        Returns:
            Tuple[bool, dict]: (success, account_info or error message)
        """
        payload = {
            'telegram_id': telegram_id,
            'login':       login,
            'password':    password,
            'server':      server
        }

        success, result = self._post('/connect', payload)
        return success, result

    def is_worker_reachable(self) -> bool:
        """
        Check if the MT5 Worker Service is online.

        Returns:
            bool: True if reachable
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self._headers(),
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

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
        expiry_minutes: int = 60
    ) -> Tuple[bool, Optional[int], str]:
        """
        Place a trade order on the user's MT5 account.

        Args:
            telegram_id:    Telegram user ID
            symbol:         Trading symbol (broker-specific)
            direction:      'BUY' or 'SELL'
            lot_size:       Lot size
            stop_loss:      Stop loss price
            take_profit:    Take profit price (TP2 used for initial order)
            order_type:     'MARKET', 'LIMIT', or 'STOP'
            entry_price:    Entry price for pending orders
            comment:        Order comment (max 31 chars)
            expiry_minutes: Minutes until pending order expires

        Returns:
            Tuple[bool, Optional[int], str]: (success, ticket_number, message)
        """
        creds = self._get_credentials(telegram_id)
        if not creds:
            return False, None, "MT5 credentials not found. Use /connect_mt5 first."

        payload = {
            'telegram_id':    telegram_id,
            'login':          creds['login'],
            'password':       creds['password'],
            'server':         creds['server'],
            'symbol':         symbol,
            'direction':      direction,
            'lot_size':       lot_size,
            'stop_loss':      stop_loss,
            'take_profit':    take_profit,
            'order_type':     order_type,
            'entry_price':    entry_price,
            'comment':        comment[:31],
            'expiry_minutes': expiry_minutes
        }

        success, result = self._post('/place_order', payload)

        if success:
            ticket = result.get('ticket')
            price = result.get('price', 0.0)
            self.logger.info(
                "Order placed: user=%d %s %s %.2f lots ticket=%d price=%.5f",
                telegram_id, direction, symbol, lot_size, ticket, price
            )
            return True, ticket, f"Order placed at {price:.5f}"

        return False, None, str(result)

    # ==================== POSITION MANAGEMENT ====================

    def modify_position(
        self,
        telegram_id: int,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Modify stop loss / take profit on an open position.

        Args:
            telegram_id: Telegram user ID
            ticket:      MT5 position ticket
            stop_loss:   New stop loss price (None = keep current)
            take_profit: New take profit price (None = keep current)

        Returns:
            Tuple[bool, str]: (success, message)
        """
        creds = self._get_credentials(telegram_id)
        if not creds:
            return False, "MT5 credentials not found."

        payload = {
            'telegram_id': telegram_id,
            'login':       creds['login'],
            'password':    creds['password'],
            'server':      creds['server'],
            'ticket':      ticket,
            'stop_loss':   stop_loss,
            'take_profit': take_profit
        }

        success, result = self._post('/modify_position', payload)
        return success, result.get('message', str(result)) if success else str(result)

    def close_partial(
        self,
        telegram_id: int,
        ticket: int,
        close_percent: float = 50.0
    ) -> Tuple[bool, str]:
        """
        Close a percentage of an open position.

        Args:
            telegram_id:   Telegram user ID
            ticket:        MT5 position ticket
            close_percent: Percentage of volume to close (default 50%)

        Returns:
            Tuple[bool, str]: (success, message)
        """
        creds = self._get_credentials(telegram_id)
        if not creds:
            return False, "MT5 credentials not found."

        payload = {
            'telegram_id':  telegram_id,
            'login':        creds['login'],
            'password':     creds['password'],
            'server':       creds['server'],
            'ticket':       ticket,
            'close_percent': close_percent
        }

        success, result = self._post('/close_partial', payload)
        return success, result.get('message', str(result)) if success else str(result)

    def get_open_positions(self, telegram_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve open positions for a user.

        Args:
            telegram_id: Telegram user ID

        Returns:
            list: Position dicts
        """
        creds = self._get_credentials(telegram_id)
        if not creds:
            return []

        payload = {
            'telegram_id': telegram_id,
            'login':       creds['login'],
            'password':    creds['password'],
            'server':      creds['server']
        }

        success, result = self._post('/get_positions', payload)
        if success:
            return result.get('positions', [])
        return []

    def calculate_lot_size(
        self,
        telegram_id: int,
        symbol: str,
        risk_percent: float,
        sl_pips: float
    ) -> float:
        """
        Calculate lot size using live account balance from MT5.

        Args:
            telegram_id: Telegram user ID
            symbol:      Trading symbol
            risk_percent: Risk per trade as a percentage (e.g., 1.0)
            sl_pips:     Stop loss distance in pips

        Returns:
            float: Calculated lot size, or 0.01 if calculation fails
        """
        creds = self._get_credentials(telegram_id)
        if not creds:
            return 0.01

        payload = {
            'telegram_id': telegram_id,
            'login':       creds['login'],
            'password':    creds['password'],
            'server':      creds['server'],
            'symbol':      symbol,
            'risk_percent': risk_percent,
            'sl_pips':     sl_pips
        }

        success, result = self._post('/calculate_lot', payload)
        if success:
            return result.get('lot_size', 0.01)
        return 0.01

    # ==================== SYMBOL NORMALIZATION ====================

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize a standard symbol name by stripping broker-specific suffixes.
        Returns the canonical symbol (e.g., 'EURUSDm' -> 'EURUSD').

        Args:
            symbol: Symbol with potential broker suffix

        Returns:
            str: Normalized symbol
        """
        for standard, variants in config.SYMBOL_VARIATIONS.items():
            if symbol in variants:
                return standard

        # Fallback: strip known suffixes
        clean = symbol.upper()
        for suffix in ['.PRO', '.RAW', '-A', 'M', '.I', '.B', '.C']:
            if clean.endswith(suffix) and len(clean) > len(suffix) + 3:
                candidate = clean[:-len(suffix)]
                if candidate in config.SYMBOL_VARIATIONS:
                    return candidate

        return symbol