import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple, Any, List

import requests

import config
import database as db

logger = logging.getLogger(__name__)

# ==================== TIMEFRAME MAPPINGS (MetaApi) ====================

_TF_MAP = {
    'M1':  '1m',  'M5':  '5m',  'M15': '15m', 'M30': '30m',
    'H1':  '1h',  'H4':  '4h',  'D1':  '1d',
    'W1':  '1w',  'MN1': '1mn',
}

_TF_MINUTES = {
    'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
    'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080, 'MN1': 43200,
}

# ==================== HTTP RETRY (Worker mode) ====================

_RETRY_DELAYS = (0, 1, 3, 9)
_WORKER_DEGRADED_COOLDOWN_SECONDS = 90
_WORKER_DEGRADED_LOG_INTERVAL_SECONDS = 60


class MT5Connector:
    """
    Dual-mode MT5 connector.

    MODE SELECTION (automatic, based on .env):
        METAAPI_TOKEN present and non-empty  ->  MetaApi cloud mode
        METAAPI_TOKEN absent or empty        ->  mt5_worker HTTP mode

    All public methods are async with identical signatures in both modes.
    To switch between modes, add or remove METAAPI_TOKEN from .env.
    No other files need to change.

    Worker mode:  POSTs to mt5_worker.py running locally or on a Windows VPS.
    MetaApi mode: uses MetaApi cloud SDK, no Windows machine required.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MT5Connector")

        self._metaapi_token = getattr(config, 'METAAPI_TOKEN', '').strip()
        self._use_metaapi   = bool(self._metaapi_token)

        # MetaApi state
        self._api:         Any            = None
        self._accounts:    Dict[int, Any] = {}
        self._connections: Dict[int, Any] = {}

        # Worker state
        self._worker_url     = getattr(
            config, 'MT5_WORKER_URL', 'http://127.0.0.1:8000').rstrip('/')
        self._worker_api_key = getattr(config, 'MT5_WORKER_API_KEY', '')
        self._worker_timeout = getattr(config, 'MT5_TIMEOUT', 30)
        self._worker_degraded_until = 0.0
        self._worker_degraded_reason = ''
        self._last_worker_degraded_log_at = 0.0
        self._market_data_telegram_id: Optional[int] = None

        if self._use_metaapi:
            self.logger.info(
                "MT5 Connector initialised. Mode: MetaApi cloud")
        else:
            self.logger.info(
                "MT5 Connector initialised. Mode: mt5_worker HTTP (%s). "
                "To use MetaApi, add METAAPI_TOKEN to .env",
                self._worker_url)

    # ================================================================
    # WORKER HTTP HELPERS
    # ================================================================

    def _worker_headers(self) -> dict:
        return {
            'Content-Type': 'application/json',
            'X-API-Key':    self._worker_api_key,
        }

    def service_label(self) -> str:
        """Human-readable label for the active MT5 connectivity backend."""
        return 'MetaApi' if self._use_metaapi else 'MT5 worker'

    def _is_worker_terminal_unavailable(self, detail: Any) -> bool:
        text = str(detail or '').lower()
        return any(marker in text for marker in (
            'mt5 terminal is not available',
            'mt5 terminal is not ready',
            'no trading account is logged in',
            'terminal not found',
        ))

    def _is_worker_degraded(self) -> bool:
        return (not self._use_metaapi) and (time.time() < self._worker_degraded_until)

    def _mark_worker_degraded(
        self,
        reason: Any,
        cooldown_seconds: int = _WORKER_DEGRADED_COOLDOWN_SECONDS,
    ) -> None:
        if self._use_metaapi:
            return
        message = str(reason or 'MT5 worker unavailable.').strip()
        now = time.time()
        was_degraded = self._is_worker_degraded()
        self._worker_degraded_until = now + max(5, int(cooldown_seconds))
        self._worker_degraded_reason = message
        if (
            not was_degraded
            or (now - self._last_worker_degraded_log_at) >= _WORKER_DEGRADED_LOG_INTERVAL_SECONDS
        ):
            self.logger.warning(
                "MT5 worker marked unavailable for %ds: %s",
                int(cooldown_seconds),
                message,
            )
            self._last_worker_degraded_log_at = now

    def _clear_worker_degraded(self) -> None:
        if self._use_metaapi:
            return
        self._worker_degraded_until = 0.0
        self._worker_degraded_reason = ''

    def last_unreachable_reason(self) -> str:
        """Best-effort explanation for why market data is currently unavailable."""
        if self._use_metaapi:
            return 'MetaApi is not reachable.'
        if self._is_worker_degraded() and self._worker_degraded_reason:
            return self._worker_degraded_reason
        return 'MT5 worker is not reachable.'

    def _worker_post(self, endpoint: str, payload: dict) -> Tuple[bool, Any]:
        url = f"{self._worker_url}{endpoint}"
        for attempt, delay in enumerate(_RETRY_DELAYS):
            if delay > 0:
                time.sleep(delay)
            try:
                resp = requests.post(
                    url, json=payload,
                    headers=self._worker_headers(),
                    timeout=(5, self._worker_timeout),
                )
                if resp.status_code == 401:
                    return False, "Worker auth failed. Check MT5_WORKER_API_KEY."
                if resp.status_code == 413:
                    return False, "Request too large."
                if resp.status_code == 503:
                    self.logger.warning(
                        "Worker busy (attempt %d/4) %s", attempt + 1, endpoint)
                    continue
                data = resp.json()
                if data.get('success'):
                    self._clear_worker_degraded()
                    return True, data
                error = data.get('error', 'Unknown worker error.')
                if self._is_worker_terminal_unavailable(error):
                    self._mark_worker_degraded(error)
                return False, error
            except requests.exceptions.ConnectionError:
                self.logger.warning(
                    "Worker unreachable at %s (attempt %d/4)", url, attempt + 1)
            except requests.exceptions.Timeout:
                self.logger.warning(
                    "Worker timeout on %s (attempt %d/4)", endpoint, attempt + 1)
            except Exception as e:
                self.logger.error("Unexpected error calling %s: %s", endpoint, e)
                return False, str(e)
        return False, "Worker unreachable after 4 attempts."

    def _worker_get(self, endpoint: str, params: dict = None) -> Tuple[bool, Any]:
        url = f"{self._worker_url}{endpoint}"
        for attempt, delay in enumerate(_RETRY_DELAYS):
            if delay > 0:
                time.sleep(delay)
            try:
                resp = requests.get(
                    url, headers=self._worker_headers(),
                    params=params or {},
                    timeout=self._worker_timeout,
                )
                if resp.status_code == 401:
                    return False, "Worker auth failed."
                if resp.status_code == 503:
                    continue
                data = resp.json()
                if data.get('success'):
                    self._clear_worker_degraded()
                    return True, data
                error = data.get('error', 'Unknown error.')
                if self._is_worker_terminal_unavailable(error):
                    self._mark_worker_degraded(error)
                return False, error
            except requests.exceptions.ConnectionError:
                self.logger.warning(
                    "Worker unreachable (GET) %s (attempt %d/4)",
                    endpoint, attempt + 1)
            except requests.exceptions.Timeout:
                self.logger.warning(
                    "Worker GET timeout %s (attempt %d/4)", endpoint, attempt + 1)
            except Exception as e:
                return False, str(e)
        return False, "Worker unreachable."

    def _get_worker_creds(self, telegram_id: int) -> Optional[dict]:
        creds = db.get_mt5_credentials(telegram_id)
        if not creds:
            self.logger.warning("No MT5 credentials for user %d", telegram_id)
        return creds

    def _get_market_data_worker_creds(self) -> Optional[dict]:
        """
        Reuse an already-connected MT5 account for shared market-data calls.
        Prefer admin accounts first, then any active subscriber with MT5 linked.
        """
        if getattr(db, 'supabase_client', None) is None or getattr(db, '_fernet', None) is None:
            return None

        candidate_ids: List[int] = []

        if self._market_data_telegram_id is not None:
            candidate_ids.append(self._market_data_telegram_id)

        try:
            users = db.get_subscribed_users()
        except Exception as e:
            self.logger.warning("Could not load subscribed users for market data: %s", e)
            users = []

        connected_ids = [
            int(user['telegram_id'])
            for user in users
            if user.get('mt5_connected') and user.get('telegram_id') is not None
        ]

        for admin_id in getattr(config, 'ADMIN_USER_IDS', []):
            candidate_ids.append(admin_id)
        candidate_ids.extend(connected_ids)

        seen: set[int] = set()
        for telegram_id in candidate_ids:
            if telegram_id in seen:
                continue
            seen.add(telegram_id)
            creds = db.get_mt5_credentials(telegram_id)
            if creds:
                self._market_data_telegram_id = telegram_id
                return creds
        return None

    def _probe_worker_health_sync(self) -> Tuple[bool, str]:
        """
        Health-check the HTTP worker with a short-lived circuit breaker.
        This prevents one MT5 outage from triggering a fan-out of candle calls.
        """
        if self._is_worker_degraded():
            return False, self.last_unreachable_reason()

        creds = self._get_market_data_worker_creds()
        if creds:
            symbol = (getattr(config, 'MONITORED_SYMBOLS', None) or ['EURUSD'])[0]
            payload = {
                'login': creds['login'],
                'password': creds['password'],
                'server': creds['server'],
                'symbol': symbol,
                'timeframe': 'M5',
                'bars': 2,
            }
            ok, res = self._worker_post('/candles', payload)
            if ok:
                self._clear_worker_degraded()
                return True, 'Credentialed MT5 market-data probe succeeded.'
            reason = str(res or 'MT5 worker market-data probe failed.')
            self._mark_worker_degraded(reason)
            return False, reason

        try:
            resp = requests.get(
                f"{self._worker_url}/health",
                headers=self._worker_headers(),
                timeout=5,
            )
            try:
                payload = resp.json()
            except Exception:
                payload = {}

            status = str(payload.get('status', '')).lower()
            message = (
                payload.get('message')
                or payload.get('error')
                or f"Health check returned HTTP {resp.status_code}."
            )

            if resp.status_code == 200 and status in {'online', 'busy', 'ok', 'healthy'}:
                self._clear_worker_degraded()
                return True, message

            cooldown = 15 if status == 'busy' else _WORKER_DEGRADED_COOLDOWN_SECONDS
            self._mark_worker_degraded(message, cooldown_seconds=cooldown)
            return False, message
        except Exception as e:
            reason = f"MT5 worker health check failed: {e}"
            self._mark_worker_degraded(reason, cooldown_seconds=30)
            return False, reason

    # ================================================================
    # METAAPI HELPERS
    # ================================================================

    async def _get_api(self):
        if self._api is None:
            try:
                from metaapi_cloud_sdk import MetaApi
                self._api = MetaApi(self._metaapi_token)
            except ImportError:
                raise RuntimeError(
                    "metaapi-cloud-sdk not installed. "
                    "Run: pip install metaapi-cloud-sdk")
        return self._api

    async def _get_connection(self, telegram_id: int):
        if telegram_id in self._connections:
            conn = self._connections[telegram_id]
            try:
                if conn.connected and conn.synchronized:
                    return conn
            except Exception:
                pass
            del self._connections[telegram_id]

        creds = db.get_mt5_credentials(telegram_id)
        if not creds:
            raise ValueError("No MT5 credentials for user %d." % telegram_id)

        metaapi_account_id = creds.get('metaapi_account_id')
        if not metaapi_account_id:
            raise ValueError(
                "No MetaApi account ID for user %d. "
                "User must reconnect via /connect_mt5." % telegram_id)

        api     = await self._get_api()
        account = await api.metatrader_account_api.get_account(metaapi_account_id)
        self._accounts[telegram_id] = account

        if account.state not in ('DEPLOYING', 'DEPLOYED'):
            await account.deploy()

        await account.wait_connected(timeout_in_seconds=120)
        connection = account.get_rpc_connection()
        await connection.connect()
        await connection.wait_synchronized(timeout_in_seconds=60)
        self._connections[telegram_id] = connection
        self.logger.info("MetaApi connection ready for user %d.", telegram_id)
        return connection

    def _to_candle_dict(self, candle: dict) -> dict:
        try:
            t = candle.get('time') or candle.get('brokerTime')
            if isinstance(t, datetime):
                ts = int(t.timestamp())
            elif isinstance(t, str):
                clean = t.replace('.000', '').replace(' ', 'T')
                if '+' not in clean and 'Z' not in clean:
                    clean += '+00:00'
                ts = int(datetime.fromisoformat(clean).timestamp())
            else:
                ts = int(t)
            return {
                'time':   ts,
                'open':   float(candle.get('open',  0)),
                'high':   float(candle.get('high',  0)),
                'low':    float(candle.get('low',   0)),
                'close':  float(candle.get('close', 0)),
                'volume': int(candle.get('tickVolume', candle.get('volume', 0))),
            }
        except Exception as e:
            self.logger.debug("Candle conversion error: %s | %s", e, candle)
            return {}

    # ================================================================
    # CONNECTIVITY
    # ================================================================

    async def is_worker_reachable(self) -> bool:
        if self._use_metaapi:
            try:
                api = await self._get_api()
                await api.metatrader_account_api.get_accounts()
                return True
            except Exception as e:
                self.logger.warning("MetaApi not reachable: %s", e)
                return False
        else:
            try:
                loop = asyncio.get_running_loop()
                ok, _reason = await loop.run_in_executor(
                    None,
                    self._probe_worker_health_sync,
                )
                return ok
            except Exception:
                return False

    def is_service_reachable_sync(self) -> bool:
        """Synchronous check for train_models.py and ml_models.py only."""
        if self._use_metaapi:
            try:
                return asyncio.run(self.is_worker_reachable())
            except Exception:
                return False
        else:
            try:
                ok, _reason = self._probe_worker_health_sync()
                return ok
            except Exception:
                return False

    # ================================================================
    # ACCOUNT VERIFICATION
    # ================================================================

    async def verify_credentials(
        self,
        telegram_id: int,
        login: int,
        password: str,
        server: str,
    ) -> Tuple[bool, Any]:
        if self._use_metaapi:
            return await self._verify_metaapi(telegram_id, login, password, server)
        else:
            return await self._verify_worker(login, password, server)

    async def _verify_worker(self, login, password, server):
        loop    = asyncio.get_running_loop()
        payload = {'login': login, 'password': password, 'server': server}
        ok, res = await loop.run_in_executor(
            None, lambda: self._worker_post('/connect', payload))
        if ok and isinstance(res, dict):
            res['metaapi_account_id'] = ''
        return ok, res

    async def _verify_metaapi(self, telegram_id, login, password, server):
        try:
            api          = await self._get_api()
            account_name = "nixie_%d" % telegram_id
            existing     = db.get_mt5_credentials(telegram_id)
            existing_id  = existing.get('metaapi_account_id') if existing else None
            account      = None

            if existing_id:
                try:
                    account = await api.metatrader_account_api.get_account(existing_id)
                except Exception:
                    account = None

            if account is None:
                account = await api.metatrader_account_api.create_account({
                    'name':     account_name,
                    'type':     'cloud',
                    'login':    str(login),
                    'password': password,
                    'server':   server,
                    'platform': 'mt5',
                    'magic':    config.MAGIC_NUMBER,
                })

            if account.state not in ('DEPLOYING', 'DEPLOYED'):
                await account.deploy()

            await account.wait_connected(timeout_in_seconds=120)
            connection = account.get_rpc_connection()
            await connection.connect()
            await connection.wait_synchronized(timeout_in_seconds=60)
            self._accounts[telegram_id]    = account
            self._connections[telegram_id] = connection

            info = await connection.get_account_information()
            return True, {
                'metaapi_account_id': account.id,
                'login':         login,
                'broker':        info.get('broker', server),
                'server':        server,
                'balance':       float(info.get('balance',    0)),
                'equity':        float(info.get('equity',     0)),
                'margin_free':   float(info.get('freeMargin', 0)),
                'currency':      info.get('currency', 'USD'),
                'leverage':      int(info.get('leverage', 100)),
                'trade_allowed': bool(info.get('tradeAllowed', True)),
            }
        except Exception as e:
            self.logger.error(
                "MetaApi verify_credentials failed user=%d: %s", telegram_id, e)
            return False, str(e)

    # ================================================================
    # MARKET DATA
    # ================================================================

    async def get_current_price(
        self, symbol: str
    ) -> Tuple[Optional[float], Optional[float]]:
        if self._use_metaapi:
            try:
                for conn in self._connections.values():
                    try:
                        if conn.connected and conn.synchronized:
                            price = await conn.get_symbol_price(symbol)
                            if price:
                                return (
                                    float(price['bid']) if price.get('bid') else None,
                                    float(price['ask']) if price.get('ask') else None,
                                )
                    except Exception:
                        continue
            except Exception as e:
                self.logger.debug("MetaApi price failed %s: %s", symbol, e)
            return None, None
        else:
            loop    = asyncio.get_running_loop()
            creds = self._get_market_data_worker_creds()
            if creds:
                payload = {
                    'login': creds['login'],
                    'password': creds['password'],
                    'server': creds['server'],
                    'symbol': symbol,
                }
                ok, res = await loop.run_in_executor(
                    None, lambda: self._worker_post('/tick', payload))
            else:
                ok, res = await loop.run_in_executor(
                    None, lambda: self._worker_get('/tick', params={'symbol': symbol}))
            if ok and isinstance(res, dict):
                return res.get('bid'), res.get('ask')
            return None, None

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        bars: int = 500,
    ) -> Optional[list]:
        if self._use_metaapi:
            return await self._candles_metaapi(symbol, timeframe, bars)
        else:
            if self._is_worker_degraded():
                return None
            loop    = asyncio.get_running_loop()
            payload = {'symbol': symbol, 'timeframe': timeframe, 'bars': bars}
            creds = self._get_market_data_worker_creds()
            if creds:
                payload.update({
                    'login': creds['login'],
                    'password': creds['password'],
                    'server': creds['server'],
                })
            ok, res = await loop.run_in_executor(
                None, lambda: self._worker_post('/candles', payload))
            if ok and isinstance(res, dict):
                self._clear_worker_degraded()
                return res.get('candles')
            if self._is_worker_terminal_unavailable(res):
                self._mark_worker_degraded(res)
                return None
            self.logger.warning(
                "Worker candles failed %s %s: %s", symbol, timeframe, res)
            return None

    async def _candles_metaapi(self, symbol: str, timeframe: str, bars: int):
        try:
            tf_upper = timeframe.upper()
            tf_ma    = _TF_MAP.get(tf_upper)
            if tf_ma is None:
                self.logger.error("Unknown timeframe: %s", timeframe)
                return None

            account = next(iter(self._accounts.values()), None)
            if account is None:
                self.logger.warning(
                    "No MetaApi account for candles %s %s.", symbol, timeframe)
                return None

            tf_mins       = _TF_MINUTES.get(tf_upper, 60)
            lookback_mins = bars * tf_mins * 1.6
            start_time    = (
                datetime.now(timezone.utc) - timedelta(minutes=lookback_mins)
            )
            candles = await account.get_historical_candles(
                symbol=symbol, timeframe=tf_ma,
                start_time=start_time, limit=bars)

            if not candles:
                return None

            result = [self._to_candle_dict(c) for c in candles]
            result = [c for c in result if c]
            result.sort(key=lambda x: x['time'])
            if len(result) > bars:
                result = result[-bars:]
            return result
        except Exception as e:
            self.logger.warning(
                "MetaApi candles failed %s %s: %s", symbol, timeframe, e)
            return None

    def get_historical_data_sync(
        self, symbol: str, timeframe: str, bars: int = 500
    ) -> Optional[list]:
        """Synchronous wrapper for train_models.py and ml_models.py ONLY."""
        if self._use_metaapi:
            try:
                return asyncio.run(self.get_historical_data(symbol, timeframe, bars))
            except Exception as e:
                self.logger.error(
                    "get_historical_data_sync failed %s %s: %s",
                    symbol, timeframe, e)
                return None
        else:
            payload = {'symbol': symbol, 'timeframe': timeframe, 'bars': bars}
            creds = self._get_market_data_worker_creds()
            if creds:
                payload.update({
                    'login': creds['login'],
                    'password': creds['password'],
                    'server': creds['server'],
                })
            ok, res = self._worker_post('/candles', payload)
            if ok and isinstance(res, dict):
                return res.get('candles')
            return None

    async def get_exchange_rates(self) -> dict:
        if self._use_metaapi:
            try:
                rates = {}
                for symbol in ['USDJPY', 'EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF']:
                    bid, ask = await self.get_current_price(symbol)
                    if bid and ask and bid > 0 and ask > 0:
                        rates[symbol] = round((bid + ask) / 2.0, 5)
                return rates
            except Exception as e:
                self.logger.error("MetaApi get_exchange_rates failed: %s", e)
                return {}
        else:
            loop    = asyncio.get_running_loop()
            creds = self._get_market_data_worker_creds()
            if creds:
                payload = {
                    'login': creds['login'],
                    'password': creds['password'],
                    'server': creds['server'],
                }
                ok, res = await loop.run_in_executor(
                    None, lambda: self._worker_post('/exchange_rates', payload))
            else:
                ok, res = await loop.run_in_executor(
                    None, lambda: self._worker_get('/exchange_rates'))
            if ok and isinstance(res, dict):
                return res.get('rates', {})
            return {}

    # ================================================================
    # ORDER EXECUTION
    # ================================================================

    async def place_order(
        self,
        telegram_id:    int,
        symbol:         str,
        direction:      str,
        lot_size:       float,
        stop_loss:      float,
        take_profit:    float,
        take_profit_2:  Optional[float] = None,
        order_type:     str = 'MARKET',
        entry_price:    Optional[float] = None,
        comment:        str = 'NIXIE TRADES',
        expiry_minutes: int = 480,
        sl_pips:        float = 0.0,
        risk_percent:   float = 1.0,
    ) -> Tuple[bool, Optional[int], float, str]:
        if self._use_metaapi:
            return await self._place_metaapi(
                telegram_id, symbol, direction, lot_size, stop_loss,
                take_profit, take_profit_2, order_type, entry_price,
                comment, expiry_minutes)
        else:
            return await self._place_worker(
                telegram_id, symbol, direction, lot_size, stop_loss,
                take_profit, take_profit_2, order_type, entry_price,
                comment, expiry_minutes, sl_pips, risk_percent)

    async def _place_worker(
        self, telegram_id, symbol, direction, lot_size, stop_loss,
        take_profit, take_profit_2, order_type, entry_price,
        comment, expiry_minutes, sl_pips, risk_percent
    ):
        creds = self._get_worker_creds(telegram_id)
        if not creds:
            return False, None, 0.0, "No MT5 credentials found."

        tp2     = take_profit_2 if take_profit_2 is not None else take_profit
        payload = {
            'login':        creds['login'],
            'password':     creds['password'],
            'server':       creds['server'],
            'symbol':       symbol,
            'direction':    direction,
            'entry':        entry_price or 0.0,
            'sl':           stop_loss,
            'tp1':          take_profit,
            'tp2':          tp2,
            'sl_pips':      sl_pips,
            'risk_percent': risk_percent,
            'lot_size':     lot_size,
            'order_type':   order_type,
            'comment':      comment[:31],
            'expiry_hours': max(1, expiry_minutes // 60),
        }

        loop    = asyncio.get_running_loop()
        ok, res = await loop.run_in_executor(
            None, lambda: self._worker_post('/execute', payload))

        if ok and isinstance(res, dict):
            ticket     = res.get('order') or res.get('deal')
            actual_lot = float(res.get('lot_size') or lot_size or 0.0)
            return True, ticket, actual_lot, "Order placed successfully."

        error_msg = res if isinstance(res, str) else "Order placement failed."
        return False, None, 0.0, error_msg

    async def _place_metaapi(
        self, telegram_id, symbol, direction, lot_size, stop_loss,
        take_profit, take_profit_2, order_type, entry_price,
        comment, expiry_minutes
    ):
        try:
            connection = await self._get_connection(telegram_id)
            tp         = take_profit_2 if take_profit_2 is not None else take_profit
            options: dict = {
                'comment': comment[:31],
                'magic':   config.MAGIC_NUMBER,
            }
            if order_type in ('LIMIT', 'STOP'):
                expiry_dt = (
                    datetime.now(timezone.utc) + timedelta(minutes=expiry_minutes)
                )
                options['expiration'] = {
                    'type': 'ORDER_TIME_SPECIFIED',
                    'time': expiry_dt,
                }

            result = None
            if direction == 'BUY':
                if order_type == 'MARKET':
                    result = await connection.create_market_buy_order(
                        symbol, lot_size, stop_loss, tp, options)
                elif order_type == 'LIMIT':
                    result = await connection.create_limit_buy_order(
                        symbol, lot_size, entry_price, stop_loss, tp, options)
                else:
                    result = await connection.create_stop_buy_order(
                        symbol, lot_size, entry_price, stop_loss, tp, options)
            else:
                if order_type == 'MARKET':
                    result = await connection.create_market_sell_order(
                        symbol, lot_size, stop_loss, tp, options)
                elif order_type == 'LIMIT':
                    result = await connection.create_limit_sell_order(
                        symbol, lot_size, entry_price, stop_loss, tp, options)
                else:
                    result = await connection.create_stop_sell_order(
                        symbol, lot_size, entry_price, stop_loss, tp, options)

            if result is None:
                return False, None, 0.0, "No result from MetaApi."

            order_id_str = str(result.get('orderId', ''))
            try:
                ticket = int(order_id_str) if order_id_str else 0
            except ValueError:
                ticket = abs(hash(order_id_str)) % (10 ** 9)

            return True, ticket, lot_size, "Order placed successfully."
        except Exception as e:
            self.logger.error(
                "MetaApi place_order failed user=%d %s %s: %s",
                telegram_id, direction, symbol, e)
            return False, None, 0.0, str(e)

    # ================================================================
    # POSITION MANAGEMENT
    # ================================================================

    async def get_positions(self, telegram_id: int) -> Tuple[bool, list]:
        if self._use_metaapi:
            return await self._positions_metaapi(telegram_id)
        else:
            return await self._positions_worker(telegram_id)

    async def _positions_worker(self, telegram_id: int):
        creds = self._get_worker_creds(telegram_id)
        if not creds:
            return False, []
        payload = {
            'login':    creds['login'],
            'password': creds['password'],
            'server':   creds['server'],
        }
        loop    = asyncio.get_running_loop()
        ok, res = await loop.run_in_executor(
            None, lambda: self._worker_post('/positions', payload))
        if ok and isinstance(res, dict):
            positions = res.get('positions', [])
            for p in positions:
                p['telegram_id'] = telegram_id
            return True, positions
        return False, []

    async def _positions_metaapi(self, telegram_id: int):
        try:
            connection    = await self._get_connection(telegram_id)
            raw_positions = await connection.get_positions()
            if raw_positions is None:
                return True, []
            result = []
            for pos in raw_positions:
                pos_id_str = str(pos.get('id', '0'))
                try:
                    ticket = int(pos_id_str)
                except ValueError:
                    ticket = abs(hash(pos_id_str)) % (10 ** 9)
                direction = 'BUY' if 'BUY' in str(pos.get('type', '')).upper() else 'SELL'
                result.append({
                    'ticket':        ticket,
                    'symbol':        pos.get('symbol', ''),
                    'type':          0 if direction == 'BUY' else 1,
                    'volume':        float(pos.get('volume',        0)),
                    'price_open':    float(pos.get('openPrice',     0)),
                    'price_current': float(pos.get('currentPrice',
                                                   pos.get('openPrice', 0))),
                    'sl':            float(pos.get('stopLoss',    0) or 0),
                    'tp':            float(pos.get('takeProfit',  0) or 0),
                    'profit':        float(pos.get('profit',      0) or 0),
                    'swap':          float(pos.get('swap',        0) or 0),
                    'time':          pos.get('time', datetime.now(timezone.utc)),
                    'time_update':   pos.get('updateTime', datetime.now(timezone.utc)),
                    'magic':         int(pos.get('magic', 0) or 0),
                    'comment':       pos.get('comment', ''),
                    '_raw_id':       pos_id_str,
                    'telegram_id':   telegram_id,
                })
            return True, result
        except Exception as e:
            self.logger.error(
                "MetaApi get_positions failed user=%d: %s", telegram_id, e)
            return False, []

    async def close_partial_position(
        self,
        telegram_id: int,
        ticket:      int,
        close_pct:   float,
    ) -> Tuple[bool, str]:
        if self._use_metaapi:
            return await self._close_partial_metaapi(telegram_id, ticket, close_pct)
        else:
            return await self._close_partial_worker(telegram_id, ticket, close_pct)

    async def _close_partial_worker(self, telegram_id, ticket, close_pct):
        creds = self._get_worker_creds(telegram_id)
        if not creds:
            return False, "No MT5 credentials."
        payload = {
            'login':     creds['login'],
            'password':  creds['password'],
            'server':    creds['server'],
            'ticket':    ticket,
            'close_pct': close_pct,
        }
        loop    = asyncio.get_running_loop()
        ok, res = await loop.run_in_executor(
            None, lambda: self._worker_post('/close_partial', payload))
        if ok:
            return True, "Closed %.0f%% of position %d." % (close_pct * 100, ticket)
        return False, res if isinstance(res, str) else "Partial close failed."

    async def _close_partial_metaapi(self, telegram_id, ticket, close_pct):
        try:
            connection = await self._get_connection(telegram_id)
            ok, positions = await self.get_positions(telegram_id)
            if not ok:
                return False, "Could not retrieve positions."
            target = next((p for p in positions if p['ticket'] == ticket), None)
            if target is None:
                return False, "Position %d not found." % ticket
            close_volume = max(0.01, round(target['volume'] * close_pct, 2))
            raw_id = target.get('_raw_id', str(ticket))
            await connection.close_position_partially(
                raw_id, close_volume, {'comment': 'NT-partial'})
            return True, "Closed %.2f lots of position %d." % (close_volume, ticket)
        except Exception as e:
            self.logger.error(
                "MetaApi close_partial failed user=%d ticket=%d: %s",
                telegram_id, ticket, e)
            return False, str(e)

    async def modify_stop_loss(
        self,
        telegram_id: int,
        ticket:      int,
        new_sl:      float,
    ) -> Tuple[bool, str]:
        if self._use_metaapi:
            return await self._modify_sl_metaapi(telegram_id, ticket, new_sl)
        else:
            return await self._modify_sl_worker(telegram_id, ticket, new_sl)

    async def _modify_sl_worker(self, telegram_id, ticket, new_sl):
        creds = self._get_worker_creds(telegram_id)
        if not creds:
            return False, "No MT5 credentials."
        payload = {
            'login':    creds['login'],
            'password': creds['password'],
            'server':   creds['server'],
            'ticket':   ticket,
            'new_sl':   new_sl,
        }
        loop    = asyncio.get_running_loop()
        ok, res = await loop.run_in_executor(
            None, lambda: self._worker_post('/modify_sl', payload))
        if ok:
            return True, "Stop loss moved to %.5f." % new_sl
        return False, res if isinstance(res, str) else "SL modification failed."

    async def _modify_sl_metaapi(self, telegram_id, ticket, new_sl):
        try:
            connection = await self._get_connection(telegram_id)
            ok, positions = await self.get_positions(telegram_id)
            if not ok:
                return False, "Could not retrieve positions."
            target = next((p for p in positions if p['ticket'] == ticket), None)
            if target is None:
                return False, "Position %d not found." % ticket
            raw_id = target.get('_raw_id', str(ticket))
            await connection.modify_position(raw_id, stop_loss=new_sl)
            return True, "Stop loss moved to %.5f." % new_sl
        except Exception as e:
            self.logger.error(
                "MetaApi modify_sl failed user=%d ticket=%d: %s",
                telegram_id, ticket, e)
            return False, str(e)

    async def get_account_info(
        self, telegram_id: int
    ) -> Tuple[bool, dict]:
        if self._use_metaapi:
            return await self._account_info_metaapi(telegram_id)
        else:
            return await self._account_info_worker(telegram_id)

    async def _account_info_worker(self, telegram_id: int):
        creds = self._get_worker_creds(telegram_id)
        if not creds:
            return False, {}
        payload = {
            'login':    creds['login'],
            'password': creds['password'],
            'server':   creds['server'],
        }
        loop    = asyncio.get_running_loop()
        ok, res = await loop.run_in_executor(
            None, lambda: self._worker_post('/account', payload))
        if ok and isinstance(res, dict):
            return True, res
        return False, {}

    async def _account_info_metaapi(self, telegram_id: int):
        try:
            connection = await self._get_connection(telegram_id)
            info       = await connection.get_account_information()
            return True, {
                'login':         info.get('login',    0),
                'broker':        info.get('broker',   ''),
                'server':        info.get('server',   ''),
                'balance':       float(info.get('balance',    0)),
                'equity':        float(info.get('equity',     0)),
                'margin':        float(info.get('margin',     0)),
                'margin_free':   float(info.get('freeMargin', 0)),
                'profit':        float(info.get('profit',     0)),
                'currency':      info.get('currency', 'USD'),
                'leverage':      int(info.get('leverage', 100)),
                'trade_allowed': bool(info.get('tradeAllowed', True)),
            }
        except Exception as e:
            self.logger.error(
                "MetaApi get_account_info failed user=%d: %s", telegram_id, e)
            return False, {}

    async def check_ticket_status(
        self, telegram_id: int, ticket: int
    ) -> dict:
        if self._use_metaapi:
            return await self._ticket_status_metaapi(telegram_id, ticket)
        else:
            return await self._ticket_status_worker(telegram_id, ticket)

    async def _ticket_status_worker(self, telegram_id: int, ticket: int):
        creds = self._get_worker_creds(telegram_id)
        loop = asyncio.get_running_loop()
        if creds:
            payload = {
                'login': creds['login'],
                'password': creds['password'],
                'server': creds['server'],
            }
            ok, res = await loop.run_in_executor(
                None, lambda: self._worker_post('/ticket_status/%d' % ticket, payload))
        else:
            ok, res = await loop.run_in_executor(
                None, lambda: self._worker_get('/ticket_status/%d' % ticket))
        if ok and isinstance(res, dict):
            return {
                'status':       res.get('status', 'UNKNOWN'),
                'close_price':  res.get('close_price'),
                'profit_pips':  res.get('profit_pips'),
                'realized_pnl': res.get('realized_pnl'),
                'closed_at':    res.get('closed_at'),
            }
        return {'status': 'UNKNOWN'}

    async def _ticket_status_metaapi(self, telegram_id: int, ticket: int):
        try:
            ok, positions = await self.get_positions(telegram_id)
            if ok:
                for pos in positions:
                    if pos['ticket'] == ticket:
                        return {'status': 'POSITION', 'ticket': ticket}

            connection = await self._get_connection(telegram_id)
            orders     = await connection.get_orders()
            if orders:
                for order in orders:
                    oid_str = str(order.get('id', ''))
                    try:
                        oid_int = int(oid_str)
                    except ValueError:
                        oid_int = abs(hash(oid_str)) % (10 ** 9)
                    if oid_int == ticket:
                        return {'status': 'PENDING', 'ticket': ticket}

            history = await connection.get_history_orders_by_ticket(str(ticket))
            if history and history.get('historyOrders'):
                last_order  = history['historyOrders'][-1]
                profit      = float(last_order.get('profit', 0) or 0)
                close_price = float(last_order.get('currentPrice', 0) or 0)
                done_time   = last_order.get('doneTime') or last_order.get('time')
                if isinstance(done_time, datetime):
                    closed_at = done_time.astimezone(timezone.utc).isoformat()
                elif done_time:
                    closed_at = str(done_time)
                else:
                    closed_at = datetime.now(timezone.utc).isoformat()
                return {
                    'status':       'CLOSED',
                    'ticket':       ticket,
                    'close_price':  close_price,
                    'profit_pips':  profit,
                    'realized_pnl': profit,
                    'closed_at':    closed_at,
                }

            return {'status': 'NOT_FOUND', 'ticket': ticket}

        except Exception as e:
            self.logger.error(
                "MetaApi check_ticket_status failed user=%d ticket=%d: %s",
                telegram_id, ticket, e)
            return {'status': 'UNKNOWN'}

    # ================================================================
    # STARTUP PRE-CONNECT
    # ================================================================

    async def connect_all_users(self):
        """
        Pre-connect all accounts at startup.
        MetaApi mode: warms up connections for instant trade execution.
        Worker mode:  does nothing — the local worker handles connections.
        """
        if not self._use_metaapi:
            self.logger.info(
                "Worker mode active. No pre-connect needed.")
            return
        try:
            all_users = db.get_subscribed_users()
            connected = [
                u for u in all_users
                if u.get('mt5_connected') and u.get('metaapi_account_id')
            ]
            if not connected:
                self.logger.info("No MetaApi accounts to pre-connect.")
                return
            self.logger.info(
                "Pre-connecting %d account(s) via MetaApi...", len(connected))
            tasks   = [self._pre_connect_user(u['telegram_id']) for u in connected]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            ok_count = sum(1 for r in results if r is True)
            self.logger.info(
                "Pre-connect complete: %d/%d accounts ready.",
                ok_count, len(connected))
        except Exception as e:
            self.logger.error("connect_all_users failed: %s", e)

    async def _pre_connect_user(self, telegram_id: int) -> bool:
        try:
            await self._get_connection(telegram_id)
            return True
        except Exception as e:
            self.logger.warning(
                "Pre-connect skipped for user %d: %s", telegram_id, e)
            return False
