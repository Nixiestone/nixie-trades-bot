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
import json
import logging
import time
from typing import Dict, Optional, Tuple

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
            provider:    'paystack', 'stripe', or 'bybit'

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
            if provider == 'stripe':
                return self._stripe_link(telegram_id, tier, amount_usd, reference)
            if provider == 'bybit':
                return self._bybit_link(telegram_id, tier, amount_usd, reference)
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
        Amount is in kobo (100 kobo = 1 USD/NGN cent).
        """
        secret = config.PAYSTACK_SECRET_KEY
        if not secret:
            self.logger.warning(
                "PAYSTACK_SECRET_KEY not set. Cannot generate Paystack link.")
            return None

        headers = {
            "Authorization": f"Bearer {secret}",
            "Content-Type":  "application/json",
        }
        payload = {
            "amount":       amount_usd * 100,
            "currency":     "USD",
            "email":        f"user{telegram_id}@nixietrades.bot",
            "reference":    reference,
            "callback_url": (
                config.PAYMENT_CALLBACK_URL or "https://t.me/NixieTradesBot"
            ),
            "metadata": {
                "telegram_id": str(telegram_id),
                "tier":        tier,
                "product":     "Nixie Trades Subscription",
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

    # ==================== BYBIT PAY ====================

    def _bybit_link(
        self,
        telegram_id: int,
        tier: str,
        amount_usd: int,
        reference: str,
    ) -> Optional[Dict]:
        """
        Create a Bybit Pay crypto checkout order.
        Falls back to the support contact URL if credentials are absent.
        """
        api_key    = config.BYBIT_API_KEY
        api_secret = config.BYBIT_API_SECRET

        if not api_key or not api_secret:
            self.logger.warning(
                "Bybit credentials not set. Directing user to support.")
            handle = config.SUPPORT_CONTACT.lstrip('@')
            return {'url': f"https://t.me/{handle}", 'reference': reference}

        merchant_id = reference[:32]
        timestamp   = str(int(time.time() * 1000))
        recv_window = "5000"
        tier_name   = TIER_DISPLAY_NAMES.get(tier, tier.capitalize())

        body = json.dumps({
            "merchantOrderId": merchant_id,
            "orderAmount":     str(amount_usd),
            "currency":        "USDT",
            "productType":     "1",
            "productName":     f"Nixie Trades {tier_name} Plan",
            "returnUrl":  (
                config.PAYMENT_SUCCESS_URL or "https://t.me/NixieTradesBot"
            ),
            "successUrl": (
                config.PAYMENT_SUCCESS_URL or "https://t.me/NixieTradesBot"
            ),
            "cancelUrl": (
                config.PAYMENT_CANCEL_URL or "https://t.me/NixieTradesBot"
            ),
        }, separators=(',', ':'))

        sign_str  = f"{timestamp}{api_key}{recv_window}{body}"
        signature = _hmac.new(
            api_secret.encode(), sign_str.encode(), hashlib.sha256
        ).hexdigest()

        headers = {
            "Content-Type":       "application/json",
            "X-BAPI-API-KEY":     api_key,
            "X-BAPI-SIGN":        signature,
            "X-BAPI-TIMESTAMP":   timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
        }

        try:
            resp = requests.post(
                "https://api.bybit.com/v3/private/pay/merchant/order/create",
                data=body, headers=headers, timeout=15,
            )
        except requests.RequestException as exc:
            self.logger.error("Bybit Pay network error: %s", exc)
            return None

        if resp.status_code != 200:
            self.logger.error(
                "Bybit Pay returned %d: %s", resp.status_code, resp.text[:200])
            return None

        data = resp.json()
        if data.get('ret_code', -1) != 0:
            self.logger.error(
                "Bybit Pay error %s: %s",
                data.get('ret_code'), data.get('ret_msg'))
            return None

        url = (
            data.get('result', {}).get('checkoutUrl')
            or data.get('result', {}).get('payUrl')
        )
        if not url:
            self.logger.error("Bybit Pay: no checkout URL in response.")
            return None

        return {'url': url, 'reference': merchant_id}

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
        return _hmac.compare_digest(expected, header_sig)

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