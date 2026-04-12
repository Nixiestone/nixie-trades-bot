"""
Nixie Trades - Payment Handler
Role: Product Manager + Quant Risk Analyst + Software Security Engineer

Subscription tier design rationale:
    FREE   — receive text setup alerts (free funnel entry)
    BASIC  — $30/month: chart images, 1 MT5 account (USD), auto-execution,
             position management, daily briefing, news alerts, /settings, /download
    PRO    — $100/month: everything Basic + 3 MT5 accounts, any account currency,
             weekly Sunday analysis
    ADMIN  — staff only, free, unlimited everything

Admin users are identified by config.ADMIN_USER_IDS and always bypass all checks.
The word 'Admin' is never shown to end users; they see 'Staff' or nothing at all.
"""

import hashlib
import hmac as _hmac
import ipaddress
import json
import logging
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import requests

import config
import database as db

logger = logging.getLogger(__name__)

# ==================== TIER CONFIGURATION ====================

# Display names shown to users.
# 'admin' is intentionally omitted from user-facing tier selection.
TIER_DISPLAY_NAMES: Dict[str, str] = {
    'free':  'Free',
    'basic': 'Basic',
    'pro':   'Pro',
}

TIER_PRICES_USD: Dict[str, int] = {
    'basic': 30,
    'pro':   100,
}

# Maximum number of MT5 accounts per tier.
TIER_ACCOUNT_LIMITS: Dict[str, int] = {
    'free':  0,
    'basic': 1,
    'pro':   3,
    'admin': 9_999,
}

# Whether the tier permits non-USD MT5 account currencies.
TIER_FOREIGN_CURRENCY: Dict[str, bool] = {
    'free':  False,
    'basic': False,   # Basic: USD accounts only
    'pro':   True,    # Pro: USD, EUR, GBP, NGN, and all others
    'admin': True,
}

# Feature → minimum tier required.
# 'free'  = any subscribed user
# 'basic' = Basic or higher
# 'pro'   = Pro or higher
# 'admin' = staff only
TIER_FEATURE_MAP: Dict[str, str] = {
    'setup_alerts_text':     'free',    # Text-only setup alerts
    'setup_alerts_chart':    'basic',   # Chart image with every alert
    'mt5_connection':        'basic',   # Connect an MT5 account
    'auto_execution':        'basic',   # Bot places trades automatically
    'position_monitoring':   'basic',   # TP/SL/breakeven management
    'settings':              'basic',   # /settings command (risk %, timezone)
    'download':              'basic',   # /download trading history CSV
    'latest_setup':          'basic',   # /latest command
    'daily_briefing':        'basic',   # 6:30 AM daily market briefing
    'news_alert':            'basic',   # 8:00 AM news summary
    'news_reminder':         'basic',   # 30-min pre-news reminders
    'weekly_analysis':       'pro',     # Sunday 9:00 AM weekly analysis
    'multi_account':         'pro',     # More than 1 MT5 account
    'foreign_currency_acct': 'pro',     # Non-USD MT5 accounts
    'admin_commands':        'admin',   # /test_scan, /test_briefing, etc.
    'ml_csv_download':       'admin',   # ML setups CSV in /download
}

# Ordered tier hierarchy (used for >= comparisons)
_TIER_ORDER: Dict[str, int] = {
    'free':  0,
    'basic': 1,
    'pro':   2,
    'admin': 3,
}

# Module-level singleton
_subscription_manager: Optional['SubscriptionManager'] = None


def get_subscription_manager() -> 'SubscriptionManager':
    """Return the module-level SubscriptionManager singleton."""
    global _subscription_manager
    if _subscription_manager is None:
        _subscription_manager = SubscriptionManager()
    return _subscription_manager


class SubscriptionManager:
    """
    Central authority for subscription tier enforcement and payment routing.

    Call pattern in bot handlers:
        sub_mgr = get_subscription_manager()
        if not sub_mgr.has_feature(telegram_id, 'settings'):
            await reply("Upgrade required...")
            return
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SubscriptionManager")
        self._usd_ngn_rate_cache: Optional[Decimal] = None
        self._usd_ngn_rate_cache_until = 0.0
        self._public_ip_cache: Optional[str] = None
        self._public_ip_cache_until = 0.0

    # ==================== TIER QUERIES ====================

    def get_tier(self, telegram_id: int) -> str:
        """
        Return the effective tier for a user.
        Admins always return 'admin' regardless of the database value.
        """
        if telegram_id in config.ADMIN_USER_IDS:
            return 'admin'
        try:
            return db.get_subscription_tier(telegram_id) or 'free'
        except Exception as exc:
            self.logger.error(
                "Could not fetch tier for user %d: %s. Defaulting to free.",
                telegram_id, exc,
            )
            return 'free'

    def has_feature(self, telegram_id: int, feature_key: str) -> bool:
        """
        Return True if the user's tier meets or exceeds the minimum required
        for `feature_key` as defined in TIER_FEATURE_MAP.
        """
        required = TIER_FEATURE_MAP.get(feature_key, 'admin')
        user_tier = self.get_tier(telegram_id)
        return _TIER_ORDER.get(user_tier, 0) >= _TIER_ORDER.get(required, 99)

    def get_account_limit(self, telegram_id: int) -> int:
        """Maximum number of MT5 accounts this user may connect."""
        return TIER_ACCOUNT_LIMITS.get(self.get_tier(telegram_id), 0)

    def can_add_account(self, telegram_id: int) -> Tuple[bool, str]:
        """
        Determine whether the user may connect another MT5 account.

        Returns:
            (True, '')                  if allowed
            (False, reason_string)      if blocked
        """
        tier  = self.get_tier(telegram_id)
        limit = TIER_ACCOUNT_LIMITS.get(tier, 0)

        if limit == 0:
            return False, (
                "Your current plan does not include MT5 account connections.\n\n"
                "Upgrade to the Basic plan ($30/month) to connect your first account "
                "and enable automated trade execution.\n"
                "Use /upgrade to continue."
            )

        try:
            current = db.get_mt5_account_count(telegram_id)
        except Exception as exc:
            self.logger.error(
                "Account count fetch failed for user %d: %s", telegram_id, exc)
            current = 0

        if current >= limit:
            if tier == 'basic':
                return False, (
                    "The Basic plan supports 1 MT5 account.\n\n"
                    "Upgrade to Pro ($100/month) to connect up to 3 accounts "
                    "in any currency.\n"
                    "Use /upgrade to change your plan."
                )
            return False, (
                f"You have reached your account limit "
                f"({current}/{limit} accounts connected).\n\n"
                "Use /upgrade to increase your limit."
            )

        return True, ''

    def can_use_foreign_currency(self, telegram_id: int) -> bool:
        """
        True if the user's plan allows non-USD MT5 account currencies.
        Basic plan: USD only.
        Pro and Admin: all currencies.
        """
        return TIER_FOREIGN_CURRENCY.get(self.get_tier(telegram_id), False)

    def upgrade_prompt(self, feature_key: str) -> str:
        """
        Return a user-facing upgrade message for a gated feature.
        Used by bot handlers when a user lacks the required tier.
        """
        required = TIER_FEATURE_MAP.get(feature_key, 'basic')
        if required == 'pro':
            plan = f"Pro plan ($100/month)"
        elif required == 'basic':
            plan = f"Basic plan ($30/month)"
        else:
            plan = "a higher plan"

        return (
            f"This feature requires the {plan}.\n\n"
            "Use /upgrade to view plans and generate a secure payment link.\n\n"
            f"{config.FOOTER}"
        )

    # ==================== PAYMENT LINK GENERATION ====================

    def generate_payment_link(
        self,
        telegram_id: int,
        tier: str,
        provider: str,
    ) -> Optional[Dict]:
        """
        Generate a checkout URL for the specified tier and payment provider.

        Args:
            telegram_id: Embedded in payment metadata for webhook processing.
            tier:        'basic' or 'pro'
            provider:    'paystack' or 'crypto' (legacy: 'bybit', 'stripe')

        Returns:
            {'url': str, 'reference': str} on success, None on failure.
        """
        if tier not in TIER_PRICES_USD:
            self.logger.error("Invalid tier '%s' for payment.", tier)
            return None

        amount_usd = TIER_PRICES_USD[tier]
        reference  = f"nixie_{telegram_id}_{tier}_{int(time.time())}"

        try:
            if provider == 'paystack':
                return self._paystack_link(telegram_id, tier, amount_usd, reference)
            if provider == 'crypto':
                return self._crypto_link(telegram_id, tier, amount_usd, reference)
            if provider == 'stripe':
                return self._stripe_link(telegram_id, tier, amount_usd, reference)
            if provider == 'bybit':
                return self._crypto_link(telegram_id, tier, amount_usd, reference)
            self.logger.error("Unknown payment provider: %s", provider)
            return None
        except Exception as exc:
            self.logger.error(
                "Payment link failed user=%d tier=%s provider=%s: %s",
                telegram_id, tier, provider, exc,
            )
            return None

    # ==================== PAYSTACK ====================

    def _paystack_link(
        self,
        telegram_id: int,
        tier: str,
        amount_usd: int,
        reference: str,
    ) -> Optional[Dict]:
        """
        Initialize a Paystack transaction.
        Amount is sent in NGN kobo (100 kobo = 1 naira).
        """
        secret = config.PAYSTACK_SECRET_KEY
        if not secret:
            self.logger.warning(
                "PAYSTACK_SECRET_KEY not set. Cannot generate Paystack link.")
            return None

        amount_subunits, display_amount = self._get_paystack_amount(amount_usd)
        headers = {
            "Authorization": f"Bearer {secret}",
            "Content-Type":  "application/json",
        }
        payload = {
            "amount":       amount_subunits,
            "currency":     "NGN",
            "email":        f"user{telegram_id}@nixietrades.bot",
            "reference":    reference,
            "callback_url": (
                config.PAYMENT_SUCCESS_URL or "https://t.me/NixieTradesBot"
            ),
            "metadata": {
                "telegram_id": str(telegram_id),
                "tier":        tier,
                "product":     "Nixie Trades Subscription",
                "amount_usd":  str(amount_usd),
                "charge_currency": "NGN",
                "charge_amount": display_amount,
            },
            "channels": ["card", "bank_transfer"],
        }

        try:
            resp = requests.post(
                "https://api.paystack.co/transaction/initialize",
                json=payload, headers=headers, timeout=15,
            )
        except requests.RequestException as exc:
            self.logger.error("Paystack network error: %s", exc)
            return None

        if resp.status_code != 200:
            self.logger.error(
                "Paystack returned %d: %s", resp.status_code, resp.text[:200])
            return None

        data = resp.json()
        if not data.get('status'):
            self.logger.error("Paystack error: %s", data.get('message'))
            return None

        return {
            'url':       data['data']['authorization_url'],
            'reference': data['data']['reference'],
        }

    def _get_paystack_amount(self, amount_usd: int) -> Tuple[int, str]:
        """Convert the USD plan price into NGN kobo using a live FX rate."""
        rate = self._get_live_usd_ngn_rate()
        major_amount = Decimal(str(amount_usd)) * rate
        subunits = self._major_to_subunits(major_amount)
        return subunits, self._format_major_amount(subunits, "NGN")

    def _get_live_usd_ngn_rate(self) -> Decimal:
        """Fetch and cache a live USD/NGN rate from public FX sources."""
        now = time.time()
        if (
            self._usd_ngn_rate_cache is not None
            and now < self._usd_ngn_rate_cache_until
        ):
            return self._usd_ngn_rate_cache

        sources = [
            (
                "open.er-api.com",
                "https://open.er-api.com/v6/latest/USD",
                lambda data: data.get("rates", {}).get("NGN"),
            ),
            (
                "exchangerate-api.com",
                "https://api.exchangerate-api.com/v4/latest/USD",
                lambda data: data.get("rates", {}).get("NGN"),
            ),
            (
                "floatrates.com",
                "https://www.floatrates.com/daily/usd.json",
                lambda data: (data.get("ngn") or {}).get("rate"),
            ),
        ]

        last_error = None
        for source_name, url, extractor in sources:
            try:
                resp = requests.get(url, timeout=8)
                resp.raise_for_status()
                data = resp.json()
                raw_rate = extractor(data)
                rate = Decimal(str(raw_rate))
                if rate <= 0 or rate >= 10000:
                    raise ValueError(f"Out-of-range USD/NGN rate: {rate}")
                self._usd_ngn_rate_cache = rate
                self._usd_ngn_rate_cache_until = now + 1800
                return rate
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "FX rate lookup failed via %s: %s", source_name, exc
                )

        if self._usd_ngn_rate_cache is not None:
            self.logger.warning(
                "Using stale cached USD/NGN rate after FX lookup failure: %s",
                last_error,
            )
            return self._usd_ngn_rate_cache

        fallback_rate = Decimal("1600")
        self._usd_ngn_rate_cache = fallback_rate
        self._usd_ngn_rate_cache_until = now + 300
        self.logger.error(
            "All FX sources failed. Falling back temporarily to USD/NGN %s: %s",
            fallback_rate,
            last_error,
        )
        return fallback_rate

    def _major_to_subunits(self, major_amount: Decimal) -> int:
        subunits = (major_amount * Decimal("100")).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
        return max(int(subunits), 100)

    def _format_major_amount(self, subunits: int, currency: str) -> str:
        major = (Decimal(subunits) / Decimal("100")).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )
        return f"{currency} {major}"

    # ==================== STRIPE ====================

    def _stripe_link(
        self,
        telegram_id: int,
        tier: str,
        amount_usd: int,
        reference: str,
    ) -> Optional[Dict]:
        """
        Create a Stripe Checkout Session for a monthly subscription.
        Uses configured Price IDs when available; falls back to inline pricing.
        """
        secret = config.STRIPE_SECRET_KEY
        if not secret:
            self.logger.warning(
                "STRIPE_SECRET_KEY not set. Cannot generate Stripe link.")
            return None

        try:
            import stripe as _stripe
        except ImportError:
            self.logger.error(
                "stripe package not installed. "
                "Run: pip install stripe --break-system-packages")
            return None

        _stripe.api_key = secret

        price_id = (
            config.STRIPE_PRICE_BASIC
            if tier == 'basic'
            else config.STRIPE_PRICE_PRO
        )

        params: dict = {
            'mode':               'subscription',
            'success_url':        (
                config.PAYMENT_SUCCESS_URL or 'https://t.me/NixieTradesBot'
            ),
            'cancel_url': (
                config.PAYMENT_CANCEL_URL or 'https://t.me/NixieTradesBot'
            ),
            'client_reference_id': str(telegram_id),
            'metadata': {
                'telegram_id': str(telegram_id),
                'tier':        tier,
                'reference':   reference,
            },
        }

        if price_id:
            params['line_items'] = [{'price': price_id, 'quantity': 1}]
        else:
            # Inline price — works without a pre-configured Stripe product
            tier_name = TIER_DISPLAY_NAMES.get(tier, tier.capitalize())
            params['line_items'] = [{
                'price_data': {
                    'currency':     'usd',
                    'unit_amount':  amount_usd * 100,
                    'product_data': {
                        'name': f"Nixie Trades {tier_name} Plan",
                        'description': (
                            f"Monthly algorithmic trading subscription "
                            f"— ${amount_usd}/month"
                        ),
                    },
                    'recurring': {'interval': 'month'},
                },
                'quantity': 1,
            }]

        try:
            session = _stripe.checkout.Session.create(**params)
            return {'url': session.url, 'reference': session.id}
        except Exception as exc:
            self.logger.error("Stripe session creation failed: %s", exc)
            return None

    # ==================== CRYPTO CHECKOUT ====================

    def _crypto_link(
        self,
        telegram_id: int,
        tier: str,
        amount_usd: int,
        reference: str,
    ) -> Optional[Dict]:
        """
        Create a crypto checkout order using the configured backend.
        The current backend uses Bybit Pay under the hood, but that
        provider name is intentionally hidden from Telegram user-facing text.
        """
        api_key      = config.BYBIT_API_KEY
        api_secret   = config.BYBIT_API_SECRET
        merchant_id  = config.BYBIT_MERCHANT_ID.strip()
        client_id    = config.BYBIT_CLIENT_ID.strip() or f"tg_{telegram_id}"
        callback_url = self._get_public_callback_url()
        source_ip    = self._detect_public_ip()

        if not api_key or not api_secret or not merchant_id:
            self.logger.warning(
                "Crypto checkout credentials incomplete. "
                "Require BYBIT_API_KEY, BYBIT_API_SECRET, and BYBIT_MERCHANT_ID. "
                "Directing user to support."
            )
            return self._crypto_fallback(reference)

        timestamp   = str(int(time.time() * 1000))
        recv_window = "5000"
        tier_name   = TIER_DISPLAY_NAMES.get(tier, tier.capitalize())
        success_url = config.PAYMENT_SUCCESS_URL or "https://t.me/NixieTradesBot"
        cancel_url  = config.PAYMENT_CANCEL_URL or "https://t.me/NixieTradesBot"

        if not callback_url:
            self.logger.warning(
                "Crypto checkout requires a public PAYMENT_CALLBACK_URL. "
                "Current value is blank, localhost, or a private host. "
                "Directing user to support."
            )
            return self._crypto_fallback(reference)

        if not source_ip:
            self.logger.warning(
                "Crypto checkout could not determine a public source IP automatically. "
                "Directing user to support."
            )
            return self._crypto_fallback(reference)

        body = json.dumps({
            "merchantId": merchant_id,
            "merchantTradeNo": reference,
            "clientId": client_id,
            "paymentType": "E_COMMERCE",
            "currency": "USDT",
            "currencyType": "crypto",
            "orderAmount": str(amount_usd),
            "goods": [{
                "goodsName": "Nixie Trades %s Plan" % tier_name,
                "goodsDetail": "Monthly %s subscription" % tier_name,
            }],
            "successUrl": success_url,
            "failedUrl": cancel_url,
            "webhookUrl": callback_url,
            "orderExpireTime": time.strftime(
                '%Y-%m-%dT%H:%M:%SZ',
                time.gmtime(int(time.time()) + 3600)
            ),
            "remark": "Nixie Trades %s monthly subscription" % tier_name,
            "env": {
                "terminalType": "WEB",
                "device": "Telegram Bot",
                "ip": source_ip,
            },
        }, separators=(',', ':'), sort_keys=True)

        sign_str  = f"{timestamp}{api_key}{recv_window}{body}"
        signature = _hmac.new(
            api_secret.encode(), sign_str.encode(), hashlib.sha256
        ).hexdigest()

        headers = {
            "Content-Type":       "application/json",
            "X-BAPI-SIGN-TYPE":   "2",
            "X-BAPI-API-KEY":     api_key,
            "X-BAPI-SIGN":        signature,
            "X-BAPI-TIMESTAMP":   timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
        }

        base_urls = self._crypto_base_urls()
        last_error = None
        for base_url in base_urls:
            try:
                resp = requests.post(
                    f"{base_url}/v5/bybitpay/create-pay",
                    data=body, headers=headers, timeout=15,
                )
            except requests.RequestException as exc:
                last_error = exc
                self.logger.warning(
                    "Crypto checkout network error via %s: %s", base_url, exc
                )
                continue

            if resp.status_code != 200:
                last_error = RuntimeError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                )
                self.logger.warning(
                    "Crypto checkout returned %d via %s: %s",
                    resp.status_code, base_url, resp.text[:200]
                )
                continue

            data = resp.json()
            ret_code = int(data.get('retCode', data.get('ret_code', -1)))
            if ret_code != 0:
                last_error = RuntimeError(
                    "%d: %s" % (
                        ret_code,
                        data.get('retMsg', data.get('ret_msg', 'Unknown error')),
                    )
                )
                self.logger.warning(
                    "Crypto checkout error via %s retCode=%d: %s",
                    base_url,
                    ret_code,
                    data.get('retMsg', data.get('ret_msg', '')),
                )
                continue

            url = self._extract_crypto_checkout_url(data)
            if url:
                return {'url': url, 'reference': reference}

            last_error = RuntimeError("No checkout URL in response.")
            self.logger.warning(
                "Crypto checkout returned no checkout URL via %s.", base_url
            )

        self.logger.error(
            "Crypto checkout unavailable across all configured endpoints: %s",
            last_error,
        )
        return self._crypto_fallback(reference)

    def _crypto_base_urls(self) -> list[str]:
        """Return unique crypto API base URLs in retry order."""
        urls = []
        for candidate in (
            "https://api2.bybit.com",
            "https://api.bytick.com",
        ):
            value = (candidate or "").strip().rstrip("/")
            if value and value not in urls:
                urls.append(value)
        return urls

    def _extract_crypto_checkout_url(self, payload: dict) -> Optional[str]:
        """
        Extract the checkout URL from a Bybit Pay response.
        Bybit Pay v5 returns the URL under result.checkoutUrl.
        Older sandbox responses used checkoutLink or payUrl.
        Also check the top-level payload in case the result wrapper is missing.
        """
        result = payload.get('result', {}) or {}
        for key in ('checkoutUrl', 'checkoutLink', 'payUrl', 'url'):
            value = (result.get(key) or '').strip()
            if value:
                return value
        # Fallback: search the top-level payload
        for key in ('checkoutUrl', 'checkoutLink', 'payUrl', 'url'):
            value = (payload.get(key) or '').strip()
            if value:
                return value
        return None

    def _get_public_callback_url(self) -> str:
        """Return the configured webhook URL only if it is publicly reachable."""
        candidate = (config.PAYMENT_CALLBACK_URL or '').strip()
        if not candidate:
            return ''
        if not self._is_public_url(candidate):
            return ''
        return candidate

    def _is_public_url(self, value: str) -> bool:
        """Reject localhost and private-network callback URLs."""
        try:
            parsed = urlparse(value)
            host = (parsed.hostname or '').strip().lower()
            if parsed.scheme not in ('http', 'https') or not host:
                return False
            if host in {'localhost', '127.0.0.1', '::1', '0.0.0.0'}:
                return False
            if host.endswith('.local'):
                return False
            try:
                ip = ipaddress.ip_address(host)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    return False
            except ValueError:
                pass
            return True
        except Exception:
            return False

    def _detect_public_ip(self) -> str:
        """Best-effort public IP detection for provider anti-fraud fields."""
        now = time.time()
        if self._public_ip_cache and now < self._public_ip_cache_until:
            return self._public_ip_cache

        sources = (
            "https://api.ipify.org",
            "https://api64.ipify.org",
            "https://ifconfig.me/ip",
        )
        last_error = None
        for url in sources:
            try:
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                candidate = (resp.text or '').strip()
                ip = ipaddress.ip_address(candidate)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    raise ValueError(f"Non-public IP returned: {candidate}")
                self._public_ip_cache = candidate
                self._public_ip_cache_until = now + 1800
                return candidate
            except Exception as exc:
                last_error = exc

        callback_url = self._get_public_callback_url()
        if callback_url:
            try:
                host = urlparse(callback_url).hostname or ''
                ip = ipaddress.ip_address(host)
                if not (ip.is_private or ip.is_loopback or ip.is_link_local):
                    self._public_ip_cache = host
                    self._public_ip_cache_until = now + 1800
                    return host
            except ValueError:
                pass

        self.logger.warning("Public IP detection failed for crypto checkout: %s", last_error)
        return ''

    def _crypto_fallback(self, reference: str) -> Dict:
        """Return a manual fallback destination when the crypto API is unavailable."""
        handle = config.SUPPORT_CONTACT.lstrip('@')
        return {'url': f"https://t.me/{handle}", 'reference': reference}

    # ==================== WEBHOOK VERIFICATION ====================

    def verify_paystack_signature(
        self, payload: bytes, header_sig: str
    ) -> bool:
        """Verify Paystack webhook HMAC-SHA512 signature."""
        if not config.PAYSTACK_SECRET_KEY:
            return False
        expected = _hmac.new(
            config.PAYSTACK_SECRET_KEY.encode(),
            payload,
            hashlib.sha512,
        ).hexdigest()
        return _hmac.compare_digest(expected, header_sig.lower().strip())

    def verify_stripe_webhook(
        self, payload: bytes, sig_header: str
    ) -> Optional[dict]:
        """Verify and decode a Stripe webhook event. Returns None on failure."""
        if not config.STRIPE_WEBHOOK_SECRET or not config.STRIPE_SECRET_KEY:
            return None
        try:
            import stripe as _stripe
            _stripe.api_key = config.STRIPE_SECRET_KEY
            event = _stripe.Webhook.construct_event(
                payload, sig_header, config.STRIPE_WEBHOOK_SECRET
            )
            return dict(event)
        except Exception as exc:
            self.logger.error("Stripe webhook verification failed: %s", exc)
            return None

    # ==================== SUBSCRIPTION ACTIVATION ====================

    def activate_subscription(
        self,
        telegram_id: int,
        tier: str,
        reference: str,
    ) -> bool:
        """
        Activate a subscription after a confirmed payment webhook.
        Writes tier and sets subscription_status = 'active'.

        Returns True if both database writes succeed.
        """
        try:
            db.set_subscription_tier(telegram_id, tier)
            db.update_subscription(telegram_id, 'active')
            self.logger.info(
                "Subscription activated: user=%d tier=%s ref=%s",
                telegram_id, tier, reference,
            )
            return True
        except Exception as exc:
            self.logger.error(
                "Subscription activation failed user=%d ref=%s: %s",
                telegram_id, reference, exc,
            )
            return False
