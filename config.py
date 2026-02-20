"""
NIX TRADES - Configuration Module
Production-ready constants, settings, and message templates
Role: Lead Architect + Python Developer
Fixes: Added missing import os, fixed unformatted {support_contact} placeholder in LEGAL_DISCLAIMER,
       added missing environment variable keys, corrected LOG_LEVEL to read from environment,
       removed bullet-point characters that render as encoding issues in non-unicode terminals.
NO EMOJIS - Professional code only
"""

import os

# ==================== APPLICATION INFO ====================

APP_NAME = "Nixie Trades"
BOT_USERNAME = "@NixieTradesBot"
VERSION = "1.0.0"
COMPANY_NAME = "Nixie Trades Limited"
SUPPORT_EMAIL = "support@nixietrades.com"
WEBSITE = "nixietrades.com"
SUPPORT_CONTACT = "@Nixiestone"
PRODUCT_NAME = "Nixie Trades"
WATERMARK_TEXT = "NIXIE TRADES"
TAGLINE = "Smart Money, Automated Logic"
FOOTER = "Nixie Trades | Educational Tool (Not Financial Advice)"

# ==================== TELEGRAM SETTINGS ====================

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
MAX_MESSAGE_LENGTH = 4096

# ==================== SUPABASE DATABASE ====================

SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')

# ==================== ENCRYPTION ====================

ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', '')

# ==================== NEWS API ====================

NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')

# ==================== MT5 WORKER SERVICE ====================

# The MT5 Worker runs on a separate Windows machine/VPS
# It exposes a REST API that this bot calls to execute trades
# This allows headless, multi-user trading without any terminal open
MT5_WORKER_URL = os.getenv('MT5_WORKER_URL', 'http://localhost:8000')
MT5_WORKER_API_KEY = os.getenv('MT5_WORKER_API_KEY', '')
MT5_TIMEOUT = 30  # seconds for HTTP requests to MT5 Worker
MT5_RETRY_ATTEMPTS = 3
MT5_RETRY_DELAY = 2  # seconds between retries

# Symbol normalization patterns
SYMBOL_SUFFIXES = ['.pro', '.raw', '.m', '.i', '.a', '.b', '.c', '_']

# ==================== TRADING PARAMETERS ====================

DEFAULT_RISK_PERCENT = 1.0
MIN_RISK_PERCENT = 0.1
MAX_RISK_PERCENT = 5.0

# Risk-Reward ratios
MIN_RR_RATIO = 1.5  # Minimum acceptable risk-reward ratio for any setup
TP1_MULTIPLE = 1.5  # TP1 is placed at 1.5x the SL distance
TP2_MULTIPLE = 2.5  # TP2 is placed at 2.5x the SL distance

# Breakeven settings
BREAKEVEN_BUFFER_PIPS = 5

# Order expiry
DEFAULT_ORDER_EXPIRY_HOURS = 1
LIMIT_ORDER_EXPIRY_MINUTES = 60
STOP_ORDER_EXPIRY_MINUTES = 60

# ==================== MONITORED SYMBOLS ====================

MONITORED_SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
    'NZDUSD', 'USDCHF', 'EURGBP', 'EURJPY', 'GBPJPY',
    'XAUUSD', 'XAGUSD'
]

# Supported Currency Pairs (same list, kept for compatibility)
CURRENCY_PAIRS = MONITORED_SYMBOLS

# Pip Sizes by Symbol Type
PIP_SIZES = {
    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'AUDUSD': 0.0001,
    'NZDUSD': 0.0001, 'USDCAD': 0.0001, 'EURGBP': 0.0001,
    'USDJPY': 0.01,   'EURJPY': 0.01,   'GBPJPY': 0.01,
    'USDCHF': 0.0001, 'XAUUSD': 0.10,   'XAGUSD': 0.01
}

# Symbol Variations for Normalization
SYMBOL_VARIATIONS = {
    'EURUSD': ['EURUSD', 'EURUSD.pro', 'EURUSD.raw', 'EURUSD-a', 'EURUSDm', 'EUR/USD'],
    'GBPUSD': ['GBPUSD', 'GBPUSD.pro', 'GBPUSD.raw', 'GBPUSD-a', 'GBPUSDm', 'GBP/USD'],
    'USDJPY': ['USDJPY', 'USDJPY.pro', 'USDJPY.raw', 'USDJPY-a', 'USDJPYm', 'USD/JPY'],
    'AUDUSD': ['AUDUSD', 'AUDUSD.pro', 'AUDUSD.raw', 'AUDUSD-a', 'AUDUSDm', 'AUD/USD'],
    'USDCAD': ['USDCAD', 'USDCAD.pro', 'USDCAD.raw', 'USDCAD-a', 'USDCADm', 'USD/CAD'],
    'NZDUSD': ['NZDUSD', 'NZDUSD.pro', 'NZDUSD.raw', 'NZDUSD-a', 'NZDUSDm', 'NZD/USD'],
    'USDCHF': ['USDCHF', 'USDCHF.pro', 'USDCHF.raw', 'USDCHF-a', 'USDCHFm', 'USD/CHF'],
    'EURGBP': ['EURGBP', 'EURGBP.pro', 'EURGBP.raw', 'EURGBP-a', 'EURGBPm', 'EUR/GBP'],
    'EURJPY': ['EURJPY', 'EURJPY.pro', 'EURJPY.raw', 'EURJPY-a', 'EURJPYm', 'EUR/JPY'],
    'GBPJPY': ['GBPJPY', 'GBPJPY.pro', 'GBPJPY.raw', 'GBPJPY-a', 'GBPJPYm', 'GBP/JPY'],
    'XAUUSD': ['XAUUSD', 'XAUUSDm', 'XAUUSD.pro', 'GOLD', 'GOLD.pro', 'GOLD-a', 'GOLDm'],
    'XAGUSD': ['XAGUSD', 'XAGUSDm', 'XAGUSD.pro', 'SILVER', 'SILVER.pro', 'SILVER-a']
}

# ==================== ML CONFIGURATION ====================

ML_THRESHOLD = 60
ML_LSTM_WEIGHT = 0.7
ML_XGBOOST_WEIGHT = 0.3
ML_SEQUENCE_LENGTH = 100
ML_RETRAINING_INTERVAL = 100

# ==================== RISK MANAGEMENT ====================

DEFAULT_RISK_PERCENT = 1.0
MAX_RISK_PIPS = 50
MIN_LOT_SIZE = 0.01
MAX_LOT_SIZE = 10.0
MAX_CURRENCY_EXPOSURE = 3

# ==================== SMC STRATEGY PARAMETERS ====================

VOLUME_THRESHOLD_OB = 1.5
VOLUME_THRESHOLD_IMPULSE = 2.0
INDUCEMENT_WICK_RATIO = 0.6
INDUCEMENT_MIN_PIPS = 3
INDUCEMENT_MAX_PIPS = 10
ATR_PERIOD = 14
ATR_MIN_RATIO = 0.7
ATR_MAX_RATIO = 2.0

# ==================== NEWS FILTER ====================

NEWS_PROXIMITY_MINUTES = 30

# ==================== ORDER TYPE DETECTION ====================

MARKET_ORDER_THRESHOLD_PIPS = 2
LIMIT_ORDER_THRESHOLD_PIPS = 20

# ==================== DRAWDOWN-BASED RISK ADJUSTMENT ====================

DRAWDOWN_LEVEL_1 = 3.0
DRAWDOWN_LEVEL_2 = 5.0
DRAWDOWN_LEVEL_3 = 8.0

RISK_MULTIPLIER_LEVEL_1 = 1.0
RISK_MULTIPLIER_LEVEL_2 = 0.7
RISK_MULTIPLIER_LEVEL_3 = 0.5
RISK_MULTIPLIER_HALT = 0.0

# ==================== FIBONACCI LEVELS ====================

# Fibonacci Levels for Adaptive TP2
# TP2 is placed using Fibonacci extensions based on HTF swing range
# Strong trend: 100% of HTF range, Moderate: 61.8%, Weak: 50%
FIBONACCI_LEVELS = {
    'strong_trend':    1.0,      # 100% extension of HTF swing range
    'moderate_trend':  0.618,    # 61.8% Fibonacci retracement
    'weak_trend':      0.5       # 50% of range
}

# TP2 Fibonacci calculation mode
USE_FIBONACCI_TP2 = True  # If False, uses fixed TP2_MULTIPLE instead

# ==================== TRADING SESSIONS (UTC hours) ====================

TRADING_SESSIONS = {
    'asian':       {'start': 0,  'end': 7,  'unicorn_only': True},
    'london_open': {'start': 7,  'end': 8,  'trading_disabled': True},
    'london':      {'start': 8,  'end': 16, 'all_setups': True},
    'overlap':     {'start': 13, 'end': 16, 'all_setups': True},
    'newyork':     {'start': 13, 'end': 21, 'all_setups': True},
    'offhours':    {'start': 21, 'end': 24, 'trading_disabled': True}
}

# ==================== FORBIDDEN WORDS ====================

FORBIDDEN_WORDS = [
    'signal', 'signals',
    'prediction', 'predictions', 'predict',
    'forecast', 'forecasts', 'forecasting',
    'ai prediction', 'ai predictions',
    'guaranteed win', 'guaranteed wins', 'guarantee',
    'investment advice', 'financial advice',
    'we recommend you buy', 'we recommend you sell',
    'you should buy', 'you should sell',
    'alpha generation', 'generate alpha',
    'profit guarantee', 'guaranteed profit',
    'sure thing', 'sure win',
    "can't lose", 'cannot lose', 'cant lose',
    'risk-free', 'risk free', 'no risk',
    'win rate'
]

WORD_REPLACEMENTS = {
    'signal':                  'automated setup',
    'signals':                 'automated setups',
    'prediction':              'model agreement score',
    'predictions':             'model agreement scores',
    'predict':                 'analyze',
    'forecast':                'historical confluence rating',
    'forecasts':               'historical confluence ratings',
    'forecasting':             'analyzing historical patterns',
    'ai prediction':           'model analysis',
    'ai predictions':          'model analyses',
    'guaranteed win':          'historical setup quality',
    'guaranteed wins':         'historical setup quality',
    'guarantee':               'historical data suggests',
    'we recommend you buy':    'setup parameters suggest long position (user has final decision)',
    'we recommend you sell':   'setup parameters suggest short position (user has final decision)',
    'you should buy':          'educational parameters indicate long (your decision)',
    'you should sell':         'educational parameters indicate short (your decision)',
    'alpha generation':        'historical edge identification',
    'generate alpha':          'identify historical edge',
    'profit guarantee':        'historical success rate (past performance does not guarantee future results)',
    'guaranteed profit':       'historical success rate (past performance does not guarantee future results)',
    'sure thing':              'high-confluence setup',
    'sure win':                'high-quality setup',
    "can't lose":              'favorable risk-reward',
    'cannot lose':             'favorable risk-reward',
    'cant lose':               'favorable risk-reward',
    'risk-free':               'risk-managed',
    'risk free':               'risk-managed',
    'no risk':                 'controlled risk',
    'win rate':                'historical success rate (past performance does not guarantee future results)'
}

# ==================== LEGAL DISCLAIMER ====================

LEGAL_DISCLAIMER = (
    "IMPORTANT LEGAL NOTICE - PLEASE READ CAREFULLY\n\n"
    "This is an educational tool designed to demonstrate algorithmic trading concepts "
    "using Smart Money Concepts (SMC). By subscribing, you acknowledge and agree:\n\n"
    "1. FINANCIAL ADVICE\n"
    "This tool does NOT provide investment advice, recommendations, or financial guidance. "
    "All automated setups are educational demonstrations of technical analysis patterns. "
    "You are solely responsible for all trading decisions.\n\n"
    "2. EDUCATIONAL PURPOSE ONLY\n"
    "This tool is designed for learning about algorithmic trading, market structure, and "
    "order flow analysis. Historical success rates shown represent past performance of the "
    "underlying methodology and do NOT guarantee future results.\n\n"
    "3. RISK DISCLOSURE\n"
    "Trading forex, CFDs, and leveraged instruments carries a high level of risk and may "
    "not be suitable for all investors. You may lose some or all of your initial investment.\n\n"
    "4. USER CONTROLS ACCOUNT\n"
    "You maintain full control of your MetaTrader 5 account at all times. You can "
    "disconnect auto-execution, manually close trades, or modify parameters at any time.\n\n"
    "5. NO GUARANTEED RETURNS\n"
    "There are no guaranteed returns, no profit guarantees, and no risk-free trades. "
    "Every trade carries risk. Historical performance reflects backtested results only.\n\n"
    "6. ACKNOWLEDGMENT\n"
    "By clicking 'I Understand and Accept,' you confirm that:\n"
    "- You are 18 years or older\n"
    "- You understand this is an educational tool, not financial advice\n"
    "- You accept full responsibility for all trading decisions\n"
    "- You understand the risks of leveraged trading\n"
    "- You will not hold Nixie Trades Limited liable for any trading losses\n\n"
    f"For support: {SUPPORT_CONTACT}\n\n"
    f"{FOOTER}")


# ==================== BOT MESSAGES ====================

WELCOME_MESSAGE = (
    "Welcome to {product_name}\n\n"
    "Institutional-grade algorithmic trading using Smart Money Concepts (SMC) "
    "with precision refinements for high-probability forex entries.\n\n"
    "What you get:\n"
    "- Real-time automated setups via Telegram\n"
    "- Smart Money analysis (Order Blocks, Breaker Blocks, Market Structure, e.t.c)\n"
    "- Machine learning confidence scoring\n"
    "- Automatic execution on MetaTrader 5\n"
    "- Risk management with partial profit-taking\n\n"
    "This is an educational tool. NOT financial advice.\n\n"
    "Commands:\n"
    "/subscribe - Start receiving automated setups\n"
    "/help - Learn about features and SMC concepts\n"
    "/latest - View most recent automated setup\n\n"
    "Support: {support_contact}\n\n"
    "{footer}"
)

SUBSCRIPTION_SUCCESS = (
    "Subscription activated successfully.\n\n"
    "You will receive automated setup alerts when market conditions align "
    "with Smart Money Concepts criteria.\n\n"
    "Daily market briefings will arrive at 8:00 AM UTC. \n"
    "Use /settings to set your timezone and risk. \n \n"
    "Want automatic trade execution?\n"
    "Use /connect_mt5 to link your MT5 broker account.\n\n"
    "For questions: {support_contact}\n\n"
    "{footer}"
)

HELP_MESSAGE = (
    "NIXIE TRADES - HELP GUIDE\n\n"
    "AVAILABLE COMMANDS:\n"
    "/start - Welcome message and overview\n"
    "/subscribe - Activate automated setup alerts\n"
    "/help - This help guide\n"
    "/connect_mt5 - Link your MT5 broker account\n"
    "/disconnect_mt5 - Unlink MT5 account\n"
    "/status - View subscription and trading statistics\n"
    "/latest - Get most recent automated setup\n"
    "/settings - Customize risk parameters and preferences\n"
    "/unsubscribe - Stop receiving alerts\n\n"
    "SETUP QUALITY LEVELS:\n"
    "- Unicorn Setup: Breaker Block plus Fair Value Gap overlap\n"
    "  Historical success rate: 72-78% (past performance does not guarantee future results)\n"
    "- Standard Setup: Order Block or Breaker Block only\n"
    "  Historical success rate: 58-62% (past performance does not guarantee future results)\n\n"
    "SMC CONCEPTS EXPLAINED:\n"
    "- Order Block (OB): Last opposite candle before institutional impulse move\n"
    "- Breaker Block (BB): Failed supply/demand zone now acting as support/resistance\n"
    "- Fair Value Gap (FVG): Price imbalance where no trading occurred\n"
    "- Break of Structure (BOS): Price breaks swing high/low in trend direction\n"
    "- Market Structure Shift (MSS): Potential trend reversal breakout\n"
    "- Inducement: Liquidity sweep that triggers retail traders before reversal\n\n"
    "ORDER TYPES:\n"
    "- Market Order: Entry within 2 pips of current price, executes immediately\n"
    "- Limit Order: Entry 3-20 pips away, waits for pullback, expires in 1 hour\n"
    "- Stop Order: Entry greater than 20 pips away, breakout entry, expires in 1 hour\n\n"
    "AUTO-EXECUTION FLOW:\n"
    "1. Setup generated when SMC criteria align\n"
    "2. Order placed on your MT5 account automatically\n"
    "3. Position monitored every 10 seconds\n"
    "4. TP1 hit: 50% closed, SL moved to breakeven\n"
    "5. TP2 hit: Remaining 50% closed, final results\n\n"
    "Need help? Contact {support_contact}\n\n"
    "{footer}"
)

ALREADY_SUBSCRIBED = (
    "You are already subscribed to Nixie Trades automated setup alerts.\n\n"
    "Use /status to view your subscription details and trading statistics.\n\n"
    "{footer}"
)

MT5_CONNECTED_ALREADY = (
    "You are already connected to MetaTrader 5.\n\n"
    "Broker: {broker_name}\n"
    "Account: {account_number}\n\n"
    "Use /disconnect_mt5 if you need to change broker accounts.\n\n"
    "{footer}"
)

MT5_CONNECTION_PROMPT = (
    "CONNECT YOUR METATRADER 5 ACCOUNT\n\n"
    "Send your credentials in this exact format (one per line):\n\n"
    "LOGIN: 12345678\n"
    "PASSWORD: YourPassword123\n"
    "SERVER: ICMarkets-Demo\n\n"
    "Your message will be deleted immediately after processing for security.\n"
    "Credentials are encrypted before storage.\n\n"
    "To find your server name: Open MT5 -> File -> Login -> Server dropdown.\n\n"
    "Cancel: Send /cancel\n\n"
    "{footer}"
)

MT5_CONNECTION_SUCCESS = (
    "Account connected successfully.\n\n"
    "Broker: {broker_name}\n"
    "Account: {account_number}\n"
    "Balance: {balance}\n\n"
    "Automatic execution is now enabled.\n"
    "You will receive notifications when trades are executed.\n\n"
    "{footer}"
)

MT5_DISCONNECTION_CONFIRM = (
    "Are you sure you want to disconnect MetaTrader 5?\n\n"
    "This will disable automatic trade execution. You will still receive "
    "automated setup alerts, but trades will not execute automatically.\n\n"
    "{footer}"
)

MT5_DISCONNECTION_SUCCESS = (
    "MetaTrader 5 disconnected successfully.\n\n"
    "Automatic trade execution is now disabled. You will continue to receive "
    "automated setup alerts.\n\n"
    "Use /connect_mt5 to reconnect anytime.\n\n"
    "{footer}"
)

NO_RECENT_SETUPS = (
    "No automated setups generated in the last 24 hours.\n\n"
    "Current market conditions have not aligned with Smart Money Concepts criteria. "
    "This is normal and part of disciplined trading.\n\n"
    "Quality over quantity.\n\n"
    "{footer}"
)

UNSUBSCRIBE_CONFIRM = (
    "Are you sure you want to unsubscribe?\n\n"
    "You will stop receiving automated setup alerts and daily market briefings.\n\n"
    "Your account data will be retained for 30 days. Use /subscribe to reactivate anytime.\n\n"
    "{footer}"
)

UNSUBSCRIBE_SUCCESS = (
    "Unsubscribed successfully.\n\n"
    "You will no longer receive automated setup alerts.\n\n"
    "Your data remains for 30 days if you wish to reactivate.\n\n"
    "Thank you for using Nixie Trades. For feedback: {support_contact}\n\n"
    "{footer}"
)

ERROR_MESSAGES = {
    'not_subscribed': (
        "This feature requires an active subscription.\n\n"
        "Use /subscribe to get started.\n\n"
        f"{FOOTER}"
    ),
    'general_error': (
        "Something went wrong. Please try again in a moment.\n\n"
        f"If this continues, contact {SUPPORT_CONTACT}\n\n"
        f"{FOOTER}"
    ),
    'mt5_not_connected': (
        "Your trading account is not connected.\n\n"
        "Use /connect_mt5 to enable automatic execution.\n\n"
        f"{FOOTER}"
    ),
    'invalid_format': (
        "The format you entered is not recognized.\n\n"
        "Please follow the instructions and try again.\n\n"
        f"{FOOTER}"
    ),
    'service_unavailable': (
        "The service is temporarily unavailable.\n\n"
        "Please try again in a few minutes.\n\n"
        f"{FOOTER}"
    )
}

# ==================== LOGGING CONFIGURATION ====================

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024
LOG_FILE_BACKUP_COUNT = 7

# ==================== DATABASE CONFIGURATION ====================

DB_POOL_SIZE = 10
DB_TIMEOUT_SECONDS = 30

# ==================== SCHEDULER CONFIGURATION ====================

POSITION_MONITOR_INTERVAL_SECONDS = 10
MARKET_SCAN_INTERVAL_MINUTES = 15
NEWS_UPDATE_INTERVAL_MINUTES = 15
ALERT_CHECK_INTERVAL_MINUTES = 60

# ==================== HTTP REQUEST CONFIGURATION ====================

REQUEST_TIMEOUT_SECONDS = 10
REQUEST_MAX_RETRIES = 3
REQUEST_BACKOFF_FACTOR = 2

# ==================== ENCRYPTION ====================

ENCRYPTION_ALGORITHM = 'Fernet'