"""
NIX TRADES Configuration
Production-ready constants and settings
NO EMOJIS - Professional code only
"""

import os
from typing import Dict, List

# ==================== APPLICATION INFO ====================

APP_NAME = "Nix Trades"
BOT_USERNAME = "@NixTradesBot"
VERSION = "1.0.0"
COMPANY_NAME = "Nix Trades Limited"
SUPPORT_EMAIL = "support@nixtrades.com"
WEBSITE = "nixtrades.com"

# ==================== TELEGRAM SETTINGS ====================

# Get from environment variables
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
MAX_MESSAGE_LENGTH = 4096

# ==================== SUPABASE DATABASE ====================

SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')

# ==================== ENCRYPTION ====================

ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', '')

# ==================== MT5 SETTINGS ====================

MT5_TIMEOUT = 60000  # milliseconds
MT5_RETRY_ATTEMPTS = 3
MT5_RETRY_DELAY = 2  # seconds

# Symbol normalization patterns
SYMBOL_SUFFIXES = ['.pro', '.raw', '.m', '.i', '.a', '.b', '.c', '_']

# ==================== TRADING PARAMETERS ====================

DEFAULT_RISK_PERCENT = 1.0  # 1% risk per trade
MIN_RISK_PERCENT = 0.5
MAX_RISK_PERCENT = 3.0

# Risk-Reward ratios
MIN_RR_RATIO = 1.5
TARGET_RR_TP1 = 1.5
TARGET_RR_TP2 = 2.5

# Breakeven settings
BREAKEVEN_BUFFER_PIPS = 5

# Order expiry
DEFAULT_ORDER_EXPIRY_HOURS = 1

# ==================== MONITORED SYMBOLS ====================

MONITORED_SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
    'NZDUSD', 'EURJPY', 'GBPJPY', 'AUDJPY', 'EURGBP',
    'XAUUSD', 'XAGUSD', 'BTCUSD', 'ETHUSD'
]

# ==================== PIP CALCULATIONS ====================

PIP_VALUES = {
    # Standard Forex Pairs (0.0001 = 1 pip)
    'EURUSD': 0.0001,
    'GBPUSD': 0.0001,
    'AUDUSD': 0.0001,
    'NZDUSD': 0.0001,
    'USDCAD': 0.0001,
    'USDCHF': 0.0001,
    'EURGBP': 0.0001,
    'EURAUD': 0.0001,
    'EURNZD': 0.0001,
    'EURCAD': 0.0001,
    'EURCHF': 0.0001,
    'GBPAUD': 0.0001,
    'GBPNZD': 0.0001,
    'GBPCAD': 0.0001,
    'GBPCHF': 0.0001,
    'AUDNZD': 0.0001,
    'AUDCAD': 0.0001,
    'AUDCHF': 0.0001,
    'NZDCAD': 0.0001,
    'NZDCHF': 0.0001,
    'CADCHF': 0.0001,
    
    # JPY Pairs (0.01 = 1 pip)
    'USDJPY': 0.01,
    'EURJPY': 0.01,
    'GBPJPY': 0.01,
    'AUDJPY': 0.01,
    'NZDJPY': 0.01,
    'CADJPY': 0.01,
    'CHFJPY': 0.01,
    
    # Precious Metals (0.10 = 1 pip)
    'XAUUSD': 0.10,
    'XAGUSD': 0.01,
    
    # Crypto (1.0 = 1 pip)
    'BTCUSD': 1.0,
    'ETHUSD': 0.10,
}

# Contract sizes
CONTRACT_SIZES = {
    'FOREX': 100000,  # Standard lot
    'XAUUSD': 100,    # Gold
    'XAGUSD': 5000,   # Silver
    'BTCUSD': 1,      # Bitcoin
    'ETHUSD': 1,      # Ethereum
}

# ==================== SMC STRATEGY SETTINGS ====================

# Volume confirmation thresholds
VOLUME_MULTIPLIER_OB = 1.5  # OB candle must have 1.5x average volume
VOLUME_MULTIPLIER_IMPULSE = 2.0  # Impulse must have 2x average volume

# Confirmation candle requirements
CONFIRMATION_BODY_RATIO = 0.5  # Body must be >50% of total candle
CONFIRMATION_ATTEMPTS_MAX = 3

# BOS requirements
REQUIRED_BOS_COUNT = 2  # Double BOS required for continuations

# Inducement validation
INDUCEMENT_WICK_MIN_PIPS = 2
INDUCEMENT_WICK_MAX_PIPS = 15
INDUCEMENT_BODY_CLOSE_RATIO = 0.7

# ATR settings
ATR_PERIOD = 14
ATR_MIN_RATIO = 0.7  # Skip if ATR < 0.7x average
ATR_MAX_RATIO = 2.0  # Skip if ATR > 2.0x average

# Session filters
AVOID_ASIAN_SESSION = True  # Lower liquidity
PREFER_LONDON_NY_OVERLAP = True  # Best liquidity

# Fibonacci TP2 extension
FIB_EXTENSION_LEVEL = 1.618

# Max concurrent exposure
MAX_CORRELATED_POSITIONS = 2
CORRELATION_THRESHOLD = 0.7

# ==================== ML MODEL SETTINGS ====================

ML_ENSEMBLE_MODELS = ['LSTM', 'XGBOOST']
ML_MIN_AGREEMENT_SCORE = 60  # Minimum to send setup
ML_AUTO_EXECUTE_THRESHOLD = 75  # Auto-execute if >= 75%

# Model confidence tiers
ML_TIER_PREMIUM = 75  # High confidence
ML_TIER_STANDARD = 60  # Moderate confidence
ML_TIER_DISCRETIONARY = 50  # Low confidence (optional)

# ==================== NEWS SETTINGS ====================

NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
NEWS_BLACKOUT_BEFORE_MINUTES = 30
NEWS_BLACKOUT_AFTER_MINUTES = 15
NEWS_IMPACT_FILTER = 'HIGH'  # Only HIGH impact news

# ==================== SCHEDULER SETTINGS ====================

DAILY_ALERT_TIME_UTC = '08:00'  # 8 AM UTC daily setup alert
MARKET_SCAN_INTERVAL_MINUTES = 15
POSITION_CHECK_INTERVAL_SECONDS = 10

# ==================== LEGAL & COMPLIANCE ====================

LEGAL_DISCLAIMER = """
Educational tool. Not financial advice. You control your account. Past performance does not guarantee future results.
"""

PERPETUAL_FOOTER = f"""
{APP_NAME} | Smart Money, Automated Logic
"""

SUBSCRIPTION_DISCLAIMER = """
IMPORTANT LEGAL NOTICE

By subscribing to automated setup alerts, you acknowledge:

1. EDUCATIONAL PURPOSE: This service provides educational trading setups based on Smart Money Concepts and machine learning analysis. It is NOT financial advice.

2. YOUR RESPONSIBILITY: You maintain full control of your trading account. All trading decisions are yours alone. You are responsible for understanding the risks involved.

3. NO GUARANTEES: Past performance does not guarantee future results. Historical success rates are provided for educational reference only.

4. RISK DISCLOSURE: Trading forex, gold, and cryptocurrencies carries substantial risk of loss. Only trade with capital you can afford to lose.

5. NOT A LICENSED ADVISOR: {COMPANY_NAME} is not a registered investment advisor, broker, or financial planner.

6. AUTO-EXECUTION RISK: If you connect your MT5 account for automated execution, trades will be placed automatically based on our algorithms. You can disconnect at any time.

Do you understand and accept these terms?
"""

START_MESSAGE = f"""
Welcome to {APP_NAME}

I am your automated Smart Money Concepts trading assistant. I analyze forex, gold, and cryptocurrency markets using institutional trading strategies combined with machine learning.

WHAT I DO:
- Detect high-probability trading setups using SMC methodology
- Provide automated entry, stop loss, and take profit levels
- Calculate position sizes based on your risk tolerance
- Optionally execute trades automatically via MT5 integration
- Track and manage your positions with breakeven protection

WHAT I DO NOT DO:
- I do not provide financial advice or predictions
- I do not guarantee profits or winning trades
- I do not have access to insider information

This is an educational tool. You maintain full control and responsibility for all trading decisions.

Use /help to see all available commands.

{LEGAL_DISCLAIMER}
"""

HELP_MESSAGE = f"""
AVAILABLE COMMANDS:

/start - Show welcome message
/help - Display this help menu
/subscribe - Activate automated setup alerts
/unsubscribe - Stop receiving alerts
/connect_mt5 - Link your MT5 account for auto-execution
/disconnect_mt5 - Unlink your MT5 account
/status - View subscription and trading statistics
/latest - Get the most recent automated setup
/settings - Adjust risk percentage per trade

SMART MONEY CONCEPTS EXPLAINED:

Order Block (OB): The last opposite-colored candle before a strong price move. Represents institutional buying or selling activity. Price often returns to these levels for continuation.

Breaker Block (BB): A previously respected support or resistance level that has been broken. After breaking, it often becomes the new support or resistance. High-probability reversal zones.

Fair Value Gap (FVG): A price imbalance on the chart where one candle's low is higher than two candles prior's high. Acts as a magnet for price, often filled before continuation.

Break of Structure (BOS): When price breaks a recent swing high (bullish) or swing low (bearish). Indicates trend continuation. We require DOUBLE BOS for higher confidence.

Market Structure Shift (MSS): When price breaks an internal structural level counter to the current trend. First sign of potential reversal. We wait for pullback to Order Block after MSS.

Inducement: The first pullback that sweeps liquidity (stop losses) before reversing. Also called a liquidity grab or stop hunt. We enter AFTER inducement, not before.

Unicorn Setup: When a Breaker Block overlaps with a Fair Value Gap. Highest probability setup due to double confluence. These receive priority.

Risk-Reward Ratio (R:R): The relationship between your stop loss distance and take profit distance. We target minimum 1:1.5, with TP1 at 1:1.5 and TP2 at 1:2.5.

{LEGAL_DISCLAIMER}
"""

SETTINGS_MESSAGE = """
RISK SETTINGS

Current risk per trade: {risk_percent}%

You can adjust your risk percentage from 0.5% to 3.0% of your account balance per trade.

Conservative: 0.5% - 1.0% (Recommended for beginners)
Moderate: 1.0% - 2.0% (Standard approach)
Aggressive: 2.0% - 3.0% (Higher risk, higher reward)

Examples (on $10,000 account):
- 1% risk = $100 max loss per trade
- 2% risk = $200 max loss per trade
- 3% risk = $300 max loss per trade

To change your risk, send a number between 0.5 and 3.0
Example: 1.5
"""

# ==================== ERROR MESSAGES ====================

ERROR_MESSAGES = {
    'no_mt5_connection': 'MT5 account not connected. Use /connect_mt5 to link your account.',
    'invalid_credentials': 'Invalid MT5 credentials. Please check your login, password, and server.',
    'connection_failed': 'Failed to connect to MT5. Please try again or contact support.',
    'symbol_not_found': 'Symbol not found on your broker. Please check the symbol name.',
    'insufficient_margin': 'Insufficient margin to place trade. Reduce position size or risk percentage.',
    'trade_failed': 'Failed to place trade. Please check your MT5 connection and try again.',
    'invalid_risk': 'Invalid risk percentage. Must be between 0.5% and 3.0%.',
    'no_setups': 'No automated setups available at this time.',
    'already_subscribed': 'You are already subscribed to automated setup alerts.',
    'not_subscribed': 'You are not currently subscribed. Use /subscribe to activate alerts.',
}

# ==================== SUCCESS MESSAGES ====================

SUCCESS_MESSAGES = {
    'subscribed': 'Subscription activated. You will receive automated setup alerts when high-probability opportunities are detected.',
    'unsubscribed': 'Subscription deactivated. You will no longer receive automated setup alerts.',
    'mt5_connected': 'MT5 account connected successfully. Automated trade execution is now enabled.',
    'mt5_disconnected': 'MT5 account disconnected. Automated trade execution is now disabled.',
    'settings_updated': 'Settings updated successfully.',
    'trade_opened': 'Trade opened successfully.',
    'position_closed': 'Position closed successfully.',
}

# ==================== FORBIDDEN WORD REPLACEMENTS ====================

# Legal compliance: Replace trading industry terms with educational language
FORBIDDEN_WORDS = {
    'signal': 'setup',
    'signals': 'setups',
    'call': 'educational opportunity',
    'calls': 'educational opportunities',
    'trade signal': 'automated setup',
    'win rate': 'historical success rate',
    'winning': 'successful',
    'winner': 'successful trade',
    'winners': 'successful trades',
    'profit': 'favorable outcome',
    'profits': 'favorable outcomes',
    'guaranteed': 'historically successful',
    'guarantee': 'historical pattern',
    'prediction': 'analysis',
    'predict': 'analyze',
    'hot tip': 'educational insight',
    'insider': 'institutional',
    'sure thing': 'high-probability setup',
    'can\'t lose': 'historically favorable',
    'easy money': 'educational opportunity',
    'get rich': 'potentially favorable',
    'money-maker': 'historically successful pattern',
    'investment advice': 'educational material',
    'financial advice': 'educational information',
    'buy now': 'consider entry at',
    'sell now': 'consider exit at',
}

# ==================== TIMEFRAME MAPPINGS ====================

TIMEFRAME_MAPPINGS = {
    '1m': 'M1',
    '5m': 'M5',
    '15m': 'M15',
    '30m': 'M30',
    '1h': 'H1',
    '4h': 'H4',
    '1d': 'D1',
    '1w': 'W1',
    '1M': 'MN1',
}

# ==================== SESSION TIMES (UTC) ====================

SESSIONS = {
    'ASIAN': {'start': '00:00', 'end': '09:00'},
    'LONDON': {'start': '08:00', 'end': '17:00'},
    'NEW_YORK': {'start': '13:00', 'end': '22:00'},
}

# ==================== LOGGING ====================

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'